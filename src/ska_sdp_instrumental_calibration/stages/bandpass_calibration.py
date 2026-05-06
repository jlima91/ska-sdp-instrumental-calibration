import logging
from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..numpy_processors.solvers import Solver
from ..plot import PlotGaintableFrequency
from ..xarray_processors._utils import parse_antenna
from ..xarray_processors.solver import run_solver
from ..xarray_processors.vis_filter import VisibilityFilter
from ._common import RUN_SOLVER_DOCSTRING
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import (
    PlotConfig,
    RunSolverConfig,
    VisibilityFilterConfig,
)

logger = logging.getLogger()


@ConfigurableStage(name="bandpass_calibration")
def bandpass_calibration_stage(
    _upstream_output_,
    _qa_dir_,
    run_solver_config: Annotated[
        RunSolverConfig,
        Field(
            description="Run Solver parameters",
            default_factory=RunSolverConfig,
        ),
    ],
    visibility_filters: Annotated[
        VisibilityFilterConfig,
        Field(
            description="""Visibility Filters which are used to flag the
            visibility data before calibration. These flags are not carry
            forwarded to the next stages.""",
            default_factory=VisibilityFilterConfig,
        ),
    ],
    plot_config: Annotated[
        PlotConfig,
        Field(description="Plot parameters", default_factory=PlotConfig),
    ],
    visibility_key: Annotated[
        Literal["vis", "corrected_vis"],
        Field(description="Visibility data to be used for calibration."),
    ] = "vis",
    export_gaintable: Annotated[
        bool,
        Field(description="Export intermediate gain solutions."),
    ] = True,
):
    """
    Performs Bandpass Calibration

    Parameters
    ----------
        _upstream_output_: dict
            Output from the upstream stage
        _qa_dir_ : str
            Directory path where the diagnostic QA outputs will be written.
        run_solver_config: RunSolverConfig
            {run_solver_docstring}
        visibility_filters: VisibilityFilterConfig
            CASA style Visibility filters
        plot_config: PlotConfig
            Configuration required for plotting.
            eg: {{plot_table: False, fixed_axis: False}}
        visibility_key: str
            Visibility data to be used for calibration.
        export_gaintable: bool
            Export intermediate gain solutions.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    # [TODO] if predict_vis stage is not run, obtain modelvis from data.
    _upstream_output_.add_checkpoint_key("gaintable")
    modelvis = _upstream_output_.modelvis
    initialtable = _upstream_output_.gaintable
    prefix = _upstream_output_.ms_prefix

    vis = _upstream_output_[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

    refant = run_solver_config.refant
    run_solver_config.refant = parse_antenna(
        refant, initialtable.configuration.names
    )

    solver = Solver.get_solver(**run_solver_config.model_dump())

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("bandpass"):
        call_counter_suffix = f"_{call_count}"

    filtered_vis = VisibilityFilter.filter(
        visibility_filters.model_dump(), vis
    )

    gaintable = run_solver(
        vis=filtered_vis,
        modelvis=modelvis,
        gaintable=initialtable,
        solver=solver,
    )

    if plot_config.plot_table:
        path_prefix = get_plots_path(
            _qa_dir_, f"{prefix}/bandpass{call_counter_suffix}"
        )

        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
            refant=_upstream_output_.refant,
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Bandpass",
                fixed_axis=plot_config.fixed_axis,
                plot_all_stations=True,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _qa_dir_,
            f"{prefix}/bandpass{call_counter_suffix}.gaintable.h5parm",
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    _upstream_output_["gaintable"] = gaintable
    _upstream_output_.increment_call_count("bandpass")
    return _upstream_output_


bandpass_calibration_stage.__doc__ = bandpass_calibration_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
