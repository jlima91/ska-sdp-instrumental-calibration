import logging
from typing import Annotated, Literal

import dask
from pydantic import BaseModel, Field
from ska_sdp_piper.piper.v2.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm as h5exp,
)
from ska_sdp_instrumental_calibration.stages._common import PlotConfig

from ...numpy_processors.solvers import Solver
from ...plot import PlotGaintableTime
from ...xarray_processors import parse_antenna, with_chunks
from ...xarray_processors.solver import run_solver
from .._utils import get_gaintables_path, get_plots_path

logger = logging.getLogger()


class RunSolverConfig(BaseModel):
    """
    A model describing the Runsolver Configuration passed
    to the Complex Gain Calibration stage
    """

    refant: Annotated[
        int | str,
        Field(description="Reference antenna."),
    ] = 0
    niter: Annotated[
        int,
        Field(description="Number of solver iterations."),
    ] = 50
    tol: Annotated[
        float,
        Field(description="Tolerance for solver convergence."),
    ] = 1e-6
    crosspol: Annotated[
        bool,
        Field(description="Include cross polarisations."),
    ] = False


@ConfigurableStage(name="complex_gain_calibration")
def complex_gain_calibration_stage(
    _upstream_output_,
    _output_dir_,
    run_solver_config: Annotated[
        RunSolverConfig,
        Field(
            description="""Run solver parameters""",
            default_factory=RunSolverConfig,
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
    ] = False,
):
    """
    Performs Complex Gain Calibration

    Parameters
    ----------
        _upstream_output_: dict
            Output from the upstream stage. It should contain:
                gaintable, modelvis and visibility data with key
                same as visibility_key
        _output_dir_ : str
            Directory path where the output file will be written.
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
        path_prefix = get_plots_path(_output_dir_, "complex_gain")

        freq_plotter = PlotGaintableTime(
            path_prefix=path_prefix,
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
            _output_dir_, "complex_gain.gaintable.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(h5exp.export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    _upstream_output_["gaintable"] = gaintable
    return _upstream_output_
