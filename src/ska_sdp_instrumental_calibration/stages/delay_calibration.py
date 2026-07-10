import logging
from typing import Annotated

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ska_sdp_instrumental_calibration.xarray_processors.apply import (
    apply_gaintable_to_dataset,
)

from ..data_managers.data_export import (
    export_clock_to_h5parm,
    export_gaintable_to_h5parm,
)
from ..data_managers.gaintable import reset_gaintable
from ..numpy_processors.solvers import Solver
from ..plot import PlotGaintableFrequency, plot_station_delays
from ..xarray_processors import parse_antenna
from ..xarray_processors.delay import (
    apply_delay_to_gaintable,
    calculate_delays_from_gain,
    calculate_delays_from_vis,
)
from ..xarray_processors.solver import run_solver
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import PlotConfig

logger = logging.getLogger()


@ConfigurableStage(name="delay_calibration")
def delay_calibration_stage(
    _upstream_output_,
    _qa_dir_,
    plot_config: Annotated[
        PlotConfig,
        Field(description="Plot parameters", default_factory=PlotConfig),
    ],
    oversample: Annotated[
        int,
        Field(description="Oversample rate"),
    ] = 1,
    use_k_type_solver: Annotated[
        bool, Field(description="Use K-type solver for delay calibration")
    ] = False,
    refant: Annotated[int | str, Field(description="Reference antenna")] = 0,
    niter: Annotated[
        int, Field(description="Number of solver iterations.")
    ] = 200,
    tol: Annotated[
        float,
        Field(
            description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance."""
        ),
    ] = 1e-06,
    export_gaintable: Annotated[
        bool, Field(description="Export intermediate gain solutions.")
    ] = True,
):
    """
    Extract delays from bandpass solutions for plotting

    Parameters
    __________
         _upstream_output_: dict
            Output from the upstream stage
        _qa_dir_ : str
            Directory path where the diagnostic QA outputs will be written.
        plot_config: PlotConfig
            Configuration required for plotting.
            eg: {plot_table: False, fixed_axis: False}
        oversample: int
            Oversample rate
        use_k_type_solver: bool
            Use K-type solver for delay calibration
        refant: int | str
            Reference antenna for delay calibration
        niter: int
            Number of solver iterations.
        tol: float
            Iteration stops when the fractional change in the gain solution is
            below this tolerance.
        export_gaintable: bool
            Export intermediate gain solutions
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    _upstream_output_.add_checkpoint_key("gaintable")
    vis = _upstream_output_.vis
    prefix = _upstream_output_.ms_prefix
    modelvis = _upstream_output_.modelvis
    gaintable = _upstream_output_.gaintable

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("delay"):
        call_counter_suffix = f"_{call_count}"

    refant = parse_antenna(refant, gaintable.configuration.names)

    if use_k_type_solver:
        delaytable = calculate_delays_from_vis(vis, refant)

    else:
        delay_solver = Solver.get_solver(refant=refant, niter=niter, tol=tol)
        logger.info(
            "Delay Calibration will be done with solver: %s", delay_solver
        )

        gaintable = run_solver(
            vis=vis,
            modelvis=modelvis,
            gaintable=gaintable,
            solver=delay_solver,
        ).persist()

        delaytable = calculate_delays_from_gain(
            gaintable, oversample
        ).persist()

        gaintable_without_delay = apply_delay_to_gaintable(
            gaintable, delaytable, inverse=True
        )

        _upstream_output_["gaintable"] = gaintable_without_delay
        _upstream_output_["bandpass_initialized_in_delay"] = True

    initialtable = reset_gaintable(gaintable)
    delay_corrections = apply_delay_to_gaintable(initialtable, delaytable)
    vis = apply_gaintable_to_dataset(vis, delay_corrections, inverse=True)

    if plot_config.plot_table:
        path_prefix = get_plots_path(
            _qa_dir_, f"{prefix}/delay{call_counter_suffix}"
        )

        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
            refant=refant,
        )

        _upstream_output_.add_compute_tasks(
            *freq_plotter.plot(
                delay_corrections,
                figure_title="Delay",
                fixed_axis=plot_config.fixed_axis,
            )
        )

        _upstream_output_.add_compute_tasks(
            plot_station_delays(
                delaytable,
                path_prefix,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _qa_dir_,
            f"{prefix}/delay{call_counter_suffix}.gaintable.h5parm",
        )

        delaytable_file_path = get_gaintables_path(
            _qa_dir_, f"{prefix}/delay{call_counter_suffix}.clock.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                delay_corrections, gaintable_file_path
            )
        )

        _upstream_output_.add_compute_tasks(
            export_clock_to_h5parm(delaytable, delaytable_file_path)
        )

    _upstream_output_["vis"] = vis
    _upstream_output_["delay"] = delay_corrections
    _upstream_output_["refant"] = refant
    _upstream_output_.add_calibration_table("delay")

    _upstream_output_.increment_call_count("delay")

    return _upstream_output_
