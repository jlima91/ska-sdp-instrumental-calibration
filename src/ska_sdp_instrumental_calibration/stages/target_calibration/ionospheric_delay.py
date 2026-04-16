import logging
from typing import Annotated, Optional

import numpy as np
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ...data_managers.gaintable import create_gaintable_from_visibility
from ...plot import PlotGaintableTargetIonosphere
from ...xarray_processors import with_chunks
from ...xarray_processors.ionosphere_solvers import IonosphericSolver
from .._utils import get_plots_path

logger = logging.getLogger()


@ConfigurableStage(name="ionospheric_delay")
def ionospheric_delay_stage(
    _upstream_output_,
    _qa_dir_,
    cluster_indexes: Annotated[
        Optional[list[int]],
        Field(
            description="""Array of integers assigning each antenna to a
            cluster. If None, all antennas are treated as a single cluster""",
        ),
    ] = None,
    block_diagonal: Annotated[
        bool,
        Field(
            description="""If True, solve for all clusters simultaneously
            assuming a block-diagonal system. If False, solve for each cluster
            sequentially""",
        ),
    ] = True,
    niter: Annotated[
        int,
        Field(
            description="""Number of solver iterations.""",
        ),
    ] = 10,
    tol: Annotated[
        float,
        Field(
            description="""Iteration stops when the fractional change in
            the gain solution is below this tolerance.""",
        ),
    ] = 1e-6,
    zernike_limit: Annotated[
        Optional[list[int]],
        Field(
            description="""list of Zernike index limits:
            Generate all Zernikes with n + |m| <= zernike_limit[cluster_id].
            If None, a default is used by the solver.""",
        ),
    ] = None,
    plot_table: Annotated[
        bool,
        Field(
            description="""Plot all station Phase vs Frequency""",
        ),
    ] = False,
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
    _upstream_output_ : UpstreamOutput
        Output from upstream stage
    _qa_dir_ : str, optional
        Directory path where the diagnostic QA outputs will be written.
    cluster_indexes : list or numpy.ndarray, optional
        An array of integers assigning each antenna to a specific cluster.
        If None, all antennas are treated as a single cluster (default: None).
    block_diagonal : bool, optional
        If True, the solver assumes a block-diagonal system and solves for all
        clusters simultaneously. If False, it solves for each cluster
        sequentially (default: True).
    niter : int, optional
        The maximum number of iterations for the solver (default: 10).
    tol : float, optional
        The tolerance for the fractional change in parameters that determines
        solver convergence (default: 1e-6).
    zernike_limit : list[int], optional
        list of Zernike index limits:
        Generate all Zernikes with n + |m| <= zernike_limit[cluster_id].
        If None, a default is used by the solver.
    plot_table: bool, optional
        Plot all station Phase vs Frequency (default: False).
    Returns
    -------
    UpstreamOutput
        The modified upstream_output object, with the vis attribute
        updated to include the applied ionospheric corrections.
    """

    if cluster_indexes is not None:
        cluster_indexes = np.array(cluster_indexes)

    _upstream_output_.add_checkpoint_key("gaintable")
    vis = _upstream_output_.vis
    modelvis = _upstream_output_.modelvis
    vis_chunks = _upstream_output_.chunks
    timeslice = _upstream_output_.timeslice

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

    _upstream_output_["gaintable"] = gaintable

    if plot_table:
        path_prefix = get_plots_path(_qa_dir_, "ionospheric_delay")

        freq_plotter = PlotGaintableTargetIonosphere(
            path_prefix=path_prefix,
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Ionospheric",
                fixed_axis=True,
            )
        )

    return _upstream_output_
