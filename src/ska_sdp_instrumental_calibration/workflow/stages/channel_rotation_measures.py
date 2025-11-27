import logging
from copy import deepcopy

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.plot_gaintable import (
    PlotGaintableFrequency,
)

from ...dask_wrappers.apply import apply_gaintable_to_dataset
from ...dask_wrappers.predict import predict_vis
from ...dask_wrappers.solver import run_solver
from ...data_managers.data_export import export_gaintable_to_h5parm
from ...data_managers.gaintable import reset_gaintable
from ...processing_tasks.rotation_measures import model_rotations
from ...processing_tasks.solvers import SolverFactory
from ..utils import (
    get_gaintables_path,
    get_plots_path,
    parse_reference_antenna,
    plot_bandpass_stages,
    plot_rm_station,
)
from ._common import RUN_SOLVER_COMMON, RUN_SOLVER_DOCSTRING

logger = logging.getLogger()


@ConfigurableStage(
    "generate_channel_rm",
    configuration=Configuration(
        oversample=ConfigParam(
            int,
            5,
            description="""Oversampling value used in the rotation
            calculatiosn. Note that setting this value to some higher
            integer may result in high memory usage.""",
        ),
        peak_threshold=ConfigParam(
            float,
            0.5,
            description="""Height of peak in the RM spectrum required
            for a rotation detection.""",
        ),
        refine_fit=ConfigParam(
            bool,
            True,
            description="""Whether or not to refine the RM spectrum
            peak locations with a nonlinear optimisation of
            the station RM values.""",
        ),
        visibility_key=ConfigParam(
            str,
            "vis",
            description="Visibility data to be used for calibration.",
            allowed_values=["vis", "corrected_vis"],
        ),
        plot_rm_config=NestedConfigParam(
            "Plot Parameters for rotational measures",
            plot_rm=ConfigParam(
                bool,
                False,
                description="""Plot the estimated rotational measures
                per station""",
            ),
            station=ConfigParam(
                (int, str),
                0,
                description="""Station number/name to be plotted""",
                nullable=True,
            ),
        ),
        plot_table=ConfigParam(
            bool, False, description="Plot the generated gain table"
        ),
        run_solver_config=NestedConfigParam(
            "Run Solver parameters",
            **{
                **(deepcopy(RUN_SOLVER_COMMON)),
                "solver": ConfigParam(
                    str,
                    "jones_substitution",
                    description="""Calibration algorithm to use.
                Options are:
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
                    50,
                    description="""Number of solver iterations.""",
                    nullable=False,
                ),
                "tol": ConfigParam(
                    float,
                    1e-03,
                    description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance.""",
                    nullable=False,
                ),
            },
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def generate_channel_rm_stage(
    upstream_output,
    oversample,
    peak_threshold,
    refine_fit,
    visibility_key,
    plot_rm_config,
    plot_table,
    run_solver_config,
    export_gaintable,
    _output_dir_,
):
    """
    Generates channel rotation measures

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
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
        plot_rm_config:
            Configs required for RM plots.
            eg: {{plot_rm: False, station: 0}}
            per station.
        plot_table: bool
            Plot the gaintable.
        run_solver_config: dict
            {run_solver_docstring}
        export_gaintable: bool
            Export intermediate gain solutions
        _output_dir_ : str
            Directory path where the output file will be written.
            Provided by piper.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """
    upstream_output.add_checkpoint_key("modelvis")
    upstream_output.add_checkpoint_key("gaintable")

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable
    beam_factory = upstream_output.beams_factory

    refant = run_solver_config["refant"]
    run_solver_config["refant"] = parse_reference_antenna(refant, initialtable)
    station = plot_rm_config["station"]
    plot_rm_config["station"] = parse_reference_antenna(station, initialtable)

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("channel_rm"):
        call_counter_suffix = f"_{call_count}"

    rotations = model_rotations(
        initialtable,
        peak_threshold=peak_threshold,
        refine_fit=refine_fit,
        refant=run_solver_config["refant"],
        oversample=oversample,
    )

    modelvis = predict_vis(
        vis,
        upstream_output["lsm"],
        initialtable.time.data,
        initialtable.soln_interval_slices,
        beam_factory,
        station_rm=rotations.rm_est,
    )

    if upstream_output["central_beams"] is not None:
        modelvis = apply_gaintable_to_dataset(
            modelvis, upstream_output["central_beams"], inverse=True
        )

    solver = SolverFactory.get_solver(**run_solver_config)
    empty_table = reset_gaintable(initialtable)

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=empty_table,
        solver=solver,
    )

    if plot_rm_config["plot_rm"]:
        path_prefix = get_plots_path(
            _output_dir_, f"channel_rm{call_counter_suffix}"
        )
        upstream_output.add_compute_tasks(
            plot_bandpass_stages(
                gaintable,
                initialtable,
                rotations.rm_est,
                run_solver_config["refant"],
                plot_path_prefix=path_prefix,
            ),
            plot_rm_station(
                initialtable,
                **rotations.get_plot_params_for_station(
                    plot_rm_config["station"]
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

        upstream_output.add_compute_tasks(
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

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["modelvis"] = modelvis
    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("channel_rm")

    return upstream_output


generate_channel_rm_stage.__doc__ = generate_channel_rm_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
