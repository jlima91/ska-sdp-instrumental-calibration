import logging
from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..data_managers.gaintable import reset_gaintable
from ..numpy_processors.solvers import Solver
from ..plot import (
    PlotGaintableFrequency,
    plot_bandpass_stages,
    plot_rm_station,
)
from ..xarray_processors import parse_antenna
from ..xarray_processors.apply import apply_gaintable_to_dataset
from ..xarray_processors.predict import predict_vis
from ..xarray_processors.rotation_measures import model_rotations
from ..xarray_processors.solver import run_solver
from ._common import RUN_SOLVER_DOCSTRING
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import PlotRMConfig, RunSolverConfig

logger = logging.getLogger()


@ConfigurableStage(name="generate_channel_rm", optional=True)
def generate_channel_rm_stage(
    _upstream_output_,
    _output_dir_,
    run_solver_config: Annotated[
        RunSolverConfig,
        Field(
            description="""Run solver parameters""",
            default_factory=RunSolverConfig,
        ),
    ],
    plot_rm_config: Annotated[
        PlotRMConfig,
        Field(
            description="""Plot parameters for rotational measures""",
            default_factory=PlotRMConfig,
        ),
    ],
    oversample: Annotated[
        int,
        Field(
            description="""Oversampling value used in the RM calculation.
            This determines the resolution of the phasor. Setting this value
            too high may result in high memory usage.""",
        ),
    ] = 5,
    peak_threshold: Annotated[
        float,
        Field(
            description="""Height of peak in the RM spectrum required
            for a rotation detection.""",
        ),
    ] = 0.5,
    refine_fit: Annotated[
        bool,
        Field(
            description="""Whether to refine RM spectrum peak locations
            with a nonlinear optimisation of station RM values.""",
        ),
    ] = True,
    visibility_key: Annotated[
        Literal["vis", "corrected_vis"],
        Field(
            description="""Visibility data to be used for calibration.""",
        ),
    ] = "vis",
    plot_table: Annotated[
        bool,
        Field(
            description="""Plot the generated gain table""",
        ),
    ] = False,
    export_gaintable: Annotated[
        bool,
        Field(
            description="""Export intermediate gain solutions.""",
        ),
    ] = False,
):
    """
    Estimate a Rotation Measure value for each station, re-predict
    model visibilities based on RM values, and run gaintable solver using
    input calibrator visibility and re-predicted model visibilities.

    Parameters
    ----------
        _upstream_output_: dict
            Output from the upstream stage
        _output_dir_ : str
            Directory path where the output file will be written.
        run_solver_config: RunSolverConfig
            {run_solver_docstring}
        plot_rm_config: PlotRMConfig
            Configs required for RM plots.
        oversample: int
            Oversampling value used in the rotation
            calculatiosn. Note that setting this value to some higher
            integer may result in high memory usage.
        peak_threshold: float
            Height of peak in the RM spectrum required
            for a rotation detection.
        refine_fit: bool
            Whether or not to refine the RM spectrum peak
            locations with a nonlinear optimisation
            of the station RM values.
        visibility_key: str
            Visibility data to be used for calibration.
        plot_table: bool
            Plot the gaintable.
        export_gaintable: bool
            Export intermediate gain solutions

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """
    _upstream_output_.add_checkpoint_key("modelvis")
    _upstream_output_.add_checkpoint_key("gaintable")

    vis = _upstream_output_[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

    modelvis = _upstream_output_.modelvis
    initialtable = _upstream_output_.gaintable
    beam_factory = _upstream_output_.beams_factory

    refant = run_solver_config.refant
    run_solver_config.refant = parse_antenna(
        refant, initialtable.configuration.names
    )
    station = plot_rm_config.station
    plot_rm_config.station = parse_antenna(
        station, initialtable.configuration.names
    )

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("channel_rm"):
        call_counter_suffix = f"_{call_count}"

    rotations = model_rotations(
        initialtable,
        peak_threshold=peak_threshold,
        refine_fit=refine_fit,
        refant=run_solver_config.refant,
        oversample=oversample,
    )

    modelvis = predict_vis(
        vis,
        _upstream_output_["lsm"],
        initialtable.time.data,
        initialtable.soln_interval_slices,
        beam_factory,
        station_rm=rotations.rm_est,
    )

    if _upstream_output_["central_beams"] is not None:
        modelvis = apply_gaintable_to_dataset(
            modelvis, _upstream_output_["central_beams"], inverse=True
        )

    solver = Solver.get_solver(**run_solver_config.model_dump())
    empty_table = reset_gaintable(initialtable)

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=empty_table,
        solver=solver,
    )

    if plot_rm_config.plot_rm:
        path_prefix = get_plots_path(
            _output_dir_, f"channel_rm{call_counter_suffix}"
        )
        _upstream_output_.add_compute_tasks(
            plot_bandpass_stages(
                gaintable,
                initialtable,
                rotations.rm_est,
                run_solver_config.refant,
                plot_path_prefix=path_prefix,
            ),
            plot_rm_station(
                initialtable,
                **rotations.get_plot_params_for_station(
                    plot_rm_config.station
                ),
                plot_path_prefix=path_prefix,
            ),
        )

    if plot_table:
        path_prefix = get_plots_path(
            _output_dir_, f"channel_rm{call_counter_suffix}"
        )

        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, f"channel_rm{call_counter_suffix}.gaintable.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    _upstream_output_["modelvis"] = modelvis
    _upstream_output_["gaintable"] = gaintable
    _upstream_output_.increment_call_count("channel_rm")

    return _upstream_output_


generate_channel_rm_stage.__doc__ = generate_channel_rm_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
