import logging
import os

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
from ska_sdp_instrumental_calibration.processing_tasks.calibrate import (
    target_solver,
)
from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.utils import (
    parse_reference_antenna,
)

logger = logging.getLogger()


@ConfigurableStage(
    "complex_gain_calibration",
    configuration=Configuration(
        run_solver_config=NestedConfigParam(
            "Run Solver parameters",
            solver=ConfigParam(
                str,
                "gain_substitution",
                description="""Calibration algorithm to use.
                (default="gain_substitution")
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
            normalise_gains=ConfigParam(
                str,
                None,
                description="""Normalises the gains.
                Only available when solver is "gain_substitution".
                Possible types of normalization are: "mean", "median".
                To perform no normalization, set this to ``null``.
                """,
                allowed_values=[None, "mean", "median"],
                nullable=True,
            ),
            timeslice=ConfigParam(
                float,
                None,
                description="""Defines time scale over which each gain solution
                is valid. This is used to define time axis of the GainTable.
                This parameter is interpreted as follows,

                float: this is a custom time interval in seconds.
                Input timestamps are grouped by intervals of this duration,
                and said groups are separately averaged to produce
                the output time axis.

                ``None``: match the time resolution of the input, i.e. copy
                the time axis of the input Visibility""",
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
    initial_gaintable = upstream_output.gaintable

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for complex gain calibration.")

    refant = run_solver_config["refant"]
    run_solver_config["refant"] = parse_reference_antenna(
        refant, initial_gaintable
    )

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("complex_gain"):
        call_counter_suffix = f"_{call_count}"

    gaintable = target_solver.run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initial_gaintable,
        **run_solver_config,
    )

    if export_gaintable:
        gaintable_file_path = os.path.join(
            _output_dir_, f"complex_gain{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(h5exp.export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("complex_gain")
    return upstream_output
