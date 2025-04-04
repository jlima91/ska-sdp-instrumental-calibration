from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import run_solver


@ConfigurableStage(
    "bandpass_calibration",
    configuration=Configuration(
        run_solver_config=NestedConfigParam(
            "Run Solver parameters",
            solver=ConfigParam(
                str,
                "gain_substitution",
                description="""Solver type to use. Currently any solver
                type accepted by solve_gaintable.
                Default is 'gain_substitution'.""",
                allowed_values=[
                    "gain_substitution",
                    "jones_substitution",
                    "normal_equations",
                    "normal_equations_presum",
                ],
            ),
            refant=ConfigParam(
                int, 0, description="""Reference antenna (defaults to 0)."""
            ),
            niter=ConfigParam(
                int,
                50,
                description="""Number of solver iterations (defaults to 50)""",
            ),
        ),
        flagging=ConfigParam(bool, False, description="Run RFI flagging"),
    ),
)
def bandpass_calibration_stage(upstream_output, run_solver_config, flagging):
    """
    Performs Bandpass Calibration

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        run_solver_config: dict
            Configuration required for bandpass calibration.
            eg: {solver: "gain_substitution", refant: 0, niter: 50}
        flagging: bool
            Run Flagging for time
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    vis = upstream_output.vis

    # [TODO] if predict_vis stage is not run, obtain modelvis from data.
    modelvis = upstream_output.modelvis

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        solver=run_solver_config["solver"],
        niter=run_solver_config["niter"],
        refant=run_solver_config["refant"],
    )

    upstream_output["gaintable"] = gaintable

    return upstream_output
