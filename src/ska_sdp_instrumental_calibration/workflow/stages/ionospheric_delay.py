import logging

import dask
import numpy as np
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    get_gaintables_path,
    get_plots_path,
    plot_gaintable,
    with_chunks,
)

from ...data_managers.data_export import export_gaintable_to_h5parm
from ...processing_tasks.calibrate.ionosphere_solvers import IonosphericSolver

logger = logging.getLogger()


@ConfigurableStage(
    "ionospheric_delay",
    configuration=Configuration(
        cluster_indexes=ConfigParam(
            list,
            None,
            description=(
                "Array of integers assigning each antenna to a cluster. "
                "If None, all antennas are treated as a single cluster"
            ),
        ),
        block_diagonal=ConfigParam(
            bool,
            True,
            description=(
                "If True, solve for all clusters simultaneously assuming a "
                "block-diagonal system. If False, solve for each"
                " cluster sequentially"
            ),
            nullable=False,
        ),
        niter=ConfigParam(
            int,
            500,
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
        zernike_limit=ConfigParam(
            int,
            None,
            description=(
                "The maximum order of Zernike polynomials to use for the "
                "screen model."
            ),
        ),
        plot_table=ConfigParam(
            bool,
            False,
            description="Plot all station Phase vs Frequency",
            nullable=False,
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def ionospheric_delay_stage(
    upstream_output,
    cluster_indexes,
    block_diagonal,
    niter,
    tol,
    zernike_limit,
    plot_table,
    export_gaintable,
    _output_dir_,
):
    """
    Calculates and applies ionospheric delay corrections to visibility data.

    This function uses an IonosphericSolver to model phase screens based on
    the difference between observed visibilities and model visibilities. It
    derives a gain table representing these phase corrections and applies it
    to the visibility data. The resulting gain table can be optionally
    exported to an H5parm file.

    Parameters
    ----------
    upstream_output : UpstreamOutput
        Output from upstream stage
    cluster_indexes : list or numpy.ndarray, optional
        An array of integers assigning each antenna to a specific cluster.
        If None, all antennas are treated as a single cluster (default: None).
    block_diagonal : bool, optional
        If True, the solver assumes a block-diagonal system and solves for all
        clusters simultaneously. If False, it solves for each cluster
        sequentially (default: True).
    niter : int, optional
        The maximum number of iterations for the solver (default: 500).
    tol : float, optional
        The tolerance for the fractional change in parameters that determines
        solver convergence (default: 1e-6).
    zernike_limit : int, optional
        The maximum order of Zernike polynomials to use for the phase screen
        model. If None, a default is used by the solver (default: None).
    plot_table: bool, optional
        Plot all station Phase vs Frequency (default: False).
    export_gaintable : bool, optional
        If True, the computed gain table is saved to an H5parm file in the
        specified output directory (default: False).
    _output_dir_ : str, optional
        Directory path where the output file will be written.

    Returns
    -------
    UpstreamOutput
        The modified upstream_output object, with the vis attribute
        updated to include the applied ionospheric corrections.
    """
    if cluster_indexes is not None:
        cluster_indexes = np.array(cluster_indexes)

    upstream_output.add_checkpoint_key("vis")
    vis = upstream_output.vis
    modelvis = upstream_output.modelvis
    vis_chunks = upstream_output.chunks

    solver = IonosphericSolver(
        vis,
        modelvis,
        cluster_indexes,
        block_diagonal,
        niter,
        tol,
        zernike_limit,
    )

    gaintable = solver.solve()
    gaintable = gaintable.pipe(with_chunks, vis_chunks)

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=True)
    upstream_output["vis"] = vis

    if plot_table:
        path_prefix = get_plots_path(_output_dir_, "ionospheric_delay")

        upstream_output.add_compute_tasks(
            plot_gaintable(gaintable, path_prefix, phase_only=True)
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, "ionospheric_delay.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    return upstream_output
