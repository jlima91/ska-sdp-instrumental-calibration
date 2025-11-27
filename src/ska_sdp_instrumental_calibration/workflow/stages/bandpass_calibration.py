import logging
from copy import deepcopy

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...dask_wrappers.solver import run_solver
from ...data_managers.data_export import export_gaintable_to_h5parm
from ...processing_tasks.solvers import SolverFactory
from ..plot_gaintable import PlotGaintableFrequency
from ..utils import (
    get_gaintables_path,
    get_plots_path,
    parse_reference_antenna,
)
from ._common import RUN_SOLVER_COMMON, RUN_SOLVER_DOCSTRING

logger = logging.getLogger()


@ConfigurableStage(
    "bandpass_calibration",
    configuration=Configuration(
        run_solver_config=NestedConfigParam(
            "Run Solver parameters",
            **{
                **(deepcopy(RUN_SOLVER_COMMON)),
                "solver": ConfigParam(
                    str,
                    "gain_substitution",
                    description="""Calibration algorithm to use. Options are:
                "gain_substitution" - original substitution algorithm
                with separate solutions for each polarisation term.
                "jones_substitution" - solve antenna-based Jones matrices
                as a whole, with independent updates within each iteration.
                "normal_equations" - solve normal equations within
                each iteration formed from linearisation with respect to
                antenna-based gain and leakage terms.
                "normal_equations_presum" - same as normal_equations
                option but with an initial accumulation of visibility
                products over time and frequency for each solution
                interval. This can be much faster for large datasets
                and solution intervals.""",
                    allowed_values=[
                        "gain_substitution",
                        "jones_substitution",
                        "normal_equations",
                        "normal_equations_presum",
                    ],
                ),
                "niter": ConfigParam(
                    int,
                    200,
                    description="""Number of solver iterations.""",
                    nullable=False,
                ),
            },
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
def bandpass_calibration_stage(
    upstream_output,
    run_solver_config,
    plot_config,
    visibility_key,
    export_gaintable,
    _output_dir_,
):
    """
    Performs Bandpass Calibration

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        run_solver_config: dict
            {run_solver_docstring}
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

    # [TODO] if predict_vis stage is not run, obtain modelvis from data.
    upstream_output.add_checkpoint_key("gaintable")
    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

    refant = run_solver_config["refant"]
    run_solver_config["refant"] = parse_reference_antenna(refant, initialtable)
    solver = SolverFactory.get_solver(**run_solver_config)

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("bandpass"):
        call_counter_suffix = f"_{call_count}"

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initialtable,
        solver=solver,
    )

    if plot_config["plot_table"]:
        path_prefix = get_plots_path(
            _output_dir_, f"bandpass{call_counter_suffix}"
        )

        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
        )

        upstream_output.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Bandpass",
                fixed_axis=plot_config["fixed_axis"],
                plot_all_stations=True,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, f"bandpass{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("bandpass")
    return upstream_output


bandpass_calibration_stage.__doc__ = bandpass_calibration_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
