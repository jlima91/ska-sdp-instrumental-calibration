import logging

import numpy as np
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ....dask_wrappers.ionosphere_solvers import IonosphericSolver
from ....data_managers.gaintable import create_gaintable_from_visibility
from ...plot_gaintable import PlotGaintableTargetIonosphere
from ...utils import get_plots_path, with_chunks

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
            10,
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

    upstream_output.add_checkpoint_key("gaintable")
    vis = upstream_output.vis
    modelvis = upstream_output.modelvis
    vis_chunks = upstream_output.chunks
    timeslice = upstream_output.timeslice

    initialtable = create_gaintable_from_visibility(
        vis, timeslice, "B", skip_default_chunk=True
    )

    initialtable = initialtable.pipe(with_chunks, vis_chunks)

    gaintable = IonosphericSolver.solve(  # pylint: disable=E1121
        vis,
        modelvis,
        initialtable,
        cluster_indexes,
        block_diagonal,
        niter,
        tol,
        zernike_limit,
    )

    upstream_output["gaintable"] = gaintable

    if plot_table:
        path_prefix = get_plots_path(_output_dir_, "ionospheric_delay")

        freq_plotter = PlotGaintableTargetIonosphere(
            path_prefix=path_prefix,
        )

        upstream_output.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Ionospheric",
                fixed_axis=True,
            )
        )

    return upstream_output
