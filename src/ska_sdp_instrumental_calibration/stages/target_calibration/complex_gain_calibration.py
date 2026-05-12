import logging
from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm as h5exp,
)

from ...numpy_processors.solvers import Solver
from ...plot import PlotGaintableTime
from ...xarray_processors import parse_antenna, with_chunks
from ...xarray_processors.solver import run_solver
from .._utils import get_gaintables_path, get_plots_path
from ..configuration_models import PlotConfig, TargetRunSolverConfig

logger = logging.getLogger()


@ConfigurableStage(name="complex_gain_calibration")
def complex_gain_calibration_stage(
    _upstream_output_,
    _qa_dir_,
    run_solver_config: Annotated[
        TargetRunSolverConfig,
        Field(
            description="""Run solver parameters""",
            default_factory=TargetRunSolverConfig,
        ),
    ],
    plot_config: Annotated[
        PlotConfig,
        Field(
            description="""Plot parameters""",
            default_factory=PlotConfig,
        ),
    ],
    visibility_key: Annotated[
        Literal["vis", "corrected_vis"],
        Field(
            description="""Visibility data to be used for calibration.""",
        ),
    ] = "vis",
    export_gaintable: Annotated[
        bool,
        Field(
            description="""Export intermediate gain solutions.""",
        ),
    ] = True,
):
    """
    Performs Complex Gain Calibration

    Parameters
    ----------
        _upstream_output_: dict
            Output from the upstream stage. It should contain:
                gaintable, modelvis and visibility data with key
                same as visibility_key
        _qa_dir_ : str
            Directory path where the diagnostic QA outputs will be written.
        run_solver_config: RunSolverConfig
            Run solver config for target calibration
        plot_config: dict
            Configuration required for plotting.
            eg: {{plot_table: False, fixed_axis: False}}
        visibility_key: str
            Visibility data to be used for calibration.
        export_gaintable: bool
            Export intermediate gain solutions

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    _upstream_output_.add_checkpoint_key("gaintable")
    modelvis = _upstream_output_.modelvis
    vis_chunks = _upstream_output_.chunks
    run_solver_config = run_solver_config.model_dump()
    run_solver_config["timeslice"] = _upstream_output_.timeslice

    vis = _upstream_output_[visibility_key]
    logger.info(f"Using {visibility_key} for complex gain calibration.")

    initial_gaintable = _upstream_output_.gaintable
    initial_gaintable = initial_gaintable.pipe(with_chunks, vis_chunks)

    refant = run_solver_config["refant"]
    run_solver_config["refant"] = parse_antenna(
        refant, initial_gaintable.configuration.names
    )

    solver = Solver.get_solver(
        **run_solver_config, solver="gain_substitution", phase_only=True
    )
    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initial_gaintable,
        solver=solver,
    )

    if plot_config.plot_table:
        path_prefix = get_plots_path(_qa_dir_, "complex_gain")

        freq_plotter = PlotGaintableTime(
            path_prefix=path_prefix, refant=run_solver_config["refant"]
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Complex Gain",
                fixed_axis=plot_config.fixed_axis,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _qa_dir_, "complex_gain.gaintable.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(h5exp.export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    _upstream_output_["gaintable"] = gaintable
    _upstream_output_["calibration_purpose"] = "gains"
    return _upstream_output_
