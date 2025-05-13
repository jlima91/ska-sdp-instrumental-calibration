import os
from copy import deepcopy

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import run_solver
from ...processing_tasks.rotation_measures import model_rotations
from ..utils import plot_gaintable
from ._common import RUN_SOLVER_DOCSTRING, RUN_SOLVER_NESTED_CONFIG


@ConfigurableStage(
    "generate_channel_rm",
    configuration=Configuration(
        fchunk=ConfigParam(
            int,
            -1,
            description="""Number of frequency channels per chunk.
            If set to -1, use fchunk value from load_data""",
        ),
        peak_threshold=ConfigParam(
            float,
            0.5,
            description="""Height of peak in the RM spectrum required
            for a rotation detection.""",
        ),
        plot_table=ConfigParam(
            bool, False, description="Plot the generated gain table"
        ),
        run_solver_config=deepcopy(RUN_SOLVER_NESTED_CONFIG),
    ),
)
def generate_channel_rm_stage(
    upstream_output,
    fchunk,
    peak_threshold,
    plot_table,
    run_solver_config,
    _output_dir_,
):
    """
    Generates channel rotation measures

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        fchunk: int
            Number of frequency channels per chunk.
            If it is '-1' fchunk of load_data will be used.
        peak_threshold: float
            Height of peak in the RM spectrum required
            for a rotation detection.
        plot_table: bool
            Plot the gaintable.
        run_solver_config: dict
            {run_solver_docstring}
        _output_dir_ : str
            Directory path where the output file will be written.
            Provided by piper.

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

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("channel_rm"):
        call_counter_suffix = f"_{call_count}"

    path_prefix = os.path.join(
        _output_dir_, f"channel_rm{call_counter_suffix}"
    )
    gaintable = model_rotations(
        initialtable,
        peak_threshold=peak_threshold,
        plot_sample=plot_table,
        plot_path_prefix=path_prefix,
    )

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=gaintable,
        solver=run_solver_config["solver"],
        niter=run_solver_config["niter"],
        refant=run_solver_config["refant"],
        phase_only=run_solver_config["phase_only"],
        tol=run_solver_config["tol"],
        crosspol=run_solver_config["crosspol"],
        normalise_gains=run_solver_config["normalise_gains"],
        jones_type=run_solver_config["jones_type"],
        timeslice=run_solver_config["timeslice"],
    )

    if plot_table:
        upstream_output.add_compute_tasks(
            plot_gaintable(
                gaintable,
                path_prefix,
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            )
        )
    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("channel_rm")

    return upstream_output


generate_channel_rm_stage.__doc__ = generate_channel_rm_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
