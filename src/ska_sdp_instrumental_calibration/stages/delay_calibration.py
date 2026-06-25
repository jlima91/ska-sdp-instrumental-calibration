import logging
from typing import Annotated

import dask
import xarray
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_visibility,
)
from ska_sdp_instrumental_calibration.xarray_processors.apply import (
    apply_gaintable_to_dataset,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import (
    calibrate_polarization,
    stack_jones_coordinate,
    unstack_jones_coordinate,
)

from ..data_managers.data_export import (
    export_clock_to_h5parm,
    export_gaintable_to_h5parm,
)
from ..numpy_processors.solvers import Solver
from ..plot import PlotGaintableFrequency, plot_station_delays
from ..xarray_processors import parse_antenna
from ..xarray_processors.delay import apply_delay, calculate_delay
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import PlotConfig

logger = logging.getLogger()

POLS = ["XX", "YY"]


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
    export_gaintable: Annotated[
        bool,
        Field(description="Export intermediate gain solutions."),
    ] = True,
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
        export_gaintable: bool
            Export intermediate gain solutions
        refant: (int,str)
            Reference antenna
        niter: int
            Number of solver iterations
        tol: float
            Tolerance value for gain solution
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    _upstream_output_.add_checkpoint_key("gaintable")
    vis = _upstream_output_.vis
    modelvis = _upstream_output_.modelvis
    initialtable = create_gaintable_from_visibility(vis, "full", "B")
    prefix = _upstream_output_.ms_prefix

    refant = parse_antenna(refant, initialtable.configuration.names)
    solver = Solver.get_solver(refant=refant, niter=niter, tol=tol)

    logger.info("Delay calibration will be done with solver: %s", solver)

    gaintable = unstack_jones_coordinate(
        initialtable,
        xarray.merge(
            [
                stack_jones_coordinate(
                    calibrate_polarization(
                        pol, vis, modelvis, initialtable, solver
                    )
                )
                for pol in POLS
            ]
        ),
    )

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("delay"):
        call_counter_suffix = f"_{call_count}"

    delaytable = calculate_delay(gaintable, oversample)

    gaintable = apply_delay(initialtable, delaytable)

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
                gaintable,
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
                gaintable, gaintable_file_path
            )
        )

        _upstream_output_.add_compute_tasks(
            export_clock_to_h5parm(delaytable, delaytable_file_path)
        )

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=False)
    _upstream_output_["vis"] = vis

    _upstream_output_.increment_call_count("delay")

    return _upstream_output_
