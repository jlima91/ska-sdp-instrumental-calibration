import logging

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm as h5exp,
)

from ...numpy_processors.solvers import Solver
from ...plot import PlotGaintableTime
from ...scheduler import UpstreamOutput
from ...xarray_processors import parse_antenna, with_chunks
from ...xarray_processors.solver import run_solver
from .._utils import get_gaintables_path, get_plots_path

logger = logging.getLogger()


@ConfigurableStage(
    "complex_gain_calibration",
    configuration=Configuration(
        run_solver_config=NestedConfigParam(
            "Run Solver parameters",
            refant=ConfigParam(
                (int, str),
                0,
                description="""Reference antenna.
                Currently only activated for gain_substitution solver""",
                nullable=False,
            ),
            niter=ConfigParam(
                int,
                50,
                description="""Number of solver iterations.""",
                nullable=False,
            ),
            tol=ConfigParam(
                float,
                1e-06,
                description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance.""",
                nullable=False,
            ),
            crosspol=ConfigParam(
                bool,
                False,
                description="""Do solutions including cross polarisations
                i.e. XY, YX or RL, LR.
                Only used by "gain_substitution" solver.""",
                nullable=False,
            ),
        ),
        plot_config=NestedConfigParam(
            "Plot parameters",
            plot_table=ConfigParam(
                bool,
                False,
                description="Plot the generated gaintable",
                nullable=False,
            ),
            fixed_axis=ConfigParam(
                bool,
                False,
                description="Limit amplitude axis to [0-1]",
                nullable=False,
            ),
        ),
        visibility_key=ConfigParam(
            str,
            "vis",
            description="Visibility data to be used for calibration.",
            allowed_values=["vis", "corrected_vis"],
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def complex_gain_calibration_stage(
    upstream_output: UpstreamOutput,
    run_solver_config,
    plot_config,
    visibility_key,
    export_gaintable,
    _output_dir_,
):
    """
    Performs Complex Gain Calibration

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage. It should contain:
              gaintable, modelvis and visibility data with key
              same as visibility_key
        run_solver_config: dict
            Run solver config for target calibration
        plot_config: dict
            Configuration required for plotting.
            eg: {{plot_table: False, fixed_axis: False}}
        visibility_key: str
            Visibility data to be used for calibration.
        export_gaintable: bool
            Export intermediate gain solutions
        _output_dir_ : str
            Directory path where the output file will be written.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    upstream_output.add_checkpoint_key("gaintable")
    modelvis = upstream_output.modelvis
    vis_chunks = upstream_output.chunks
    run_solver_config["timeslice"] = upstream_output.timeslice

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for complex gain calibration.")

    initial_gaintable = upstream_output.gaintable
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

    if plot_config["plot_table"]:
        path_prefix = get_plots_path(_output_dir_, "complex_gain")

        freq_plotter = PlotGaintableTime(
            path_prefix=path_prefix,
        )

        upstream_output.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Complex Gain",
                fixed_axis=plot_config["fixed_axis"],
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, "complex_gain.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(h5exp.export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    return upstream_output
