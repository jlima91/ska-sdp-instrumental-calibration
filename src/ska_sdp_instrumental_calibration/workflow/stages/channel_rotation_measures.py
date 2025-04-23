import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import run_solver
from ...processing_tasks.post_processing import model_rotations


@ConfigurableStage(
    "generate_channel_rm",
    configuration=Configuration(
        fchunk=ConfigParam(
            int,
            -1,
            description="""Number of frequency channels per chunk.
            If set to -1, use fchunk value from load_data""",
        ),
        run_solver_config=NestedConfigParam(
            "Run Solver Parameters",
            solver=ConfigParam(
                str,
                "normal_equations",
                description="""Solver type to use. Currently any solver
                type accepted by solve_gaintable.
                Default is 'normal_equations'.""",
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
    ),
)
def generate_channel_rm_stage(upstream_output, fchunk, run_solver_config):
    """
    Generates channel rotation measures

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        fchunk: int
            Number of frequency channels per chunk.
            If it is '-1' fchunk of load_data will be used.
        run_solver_config: dict
            Configuration required for bandpass calibration.
            eg: {solver: "gain_substitution", refant: 0, niter: 50}
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """
    vis = upstream_output.vis
    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable
    if fchunk != -1:
        initialtable = upstream_output.gaintable.chunk({"frequency": fchunk})

    gaintable = dask.delayed(model_rotations)(initialtable, plot_sample=True)

    gaintable = dask.delayed(run_solver)(
        vis=vis,
        modelvis=modelvis,
        gaintable=gaintable,
        solver=run_solver_config["solver"],
        niter=run_solver_config["niter"],
        refant=run_solver_config["refant"],
    )

    upstream_output["gaintable"] = gaintable
    return upstream_output
