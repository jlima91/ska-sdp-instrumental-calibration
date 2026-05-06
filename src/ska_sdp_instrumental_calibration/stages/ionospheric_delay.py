import logging
from typing import Annotated, Literal, Optional, Union

import dask
import numpy as np
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..data_managers.gaintable import create_gaintable_from_visibility
from ..plot import PlotGaintableFrequency
from ..xarray_processors import with_chunks
from ..xarray_processors.apply import apply_gaintable_to_dataset
from ..xarray_processors.ionosphere_solvers import IonosphericSolver
from ._utils import get_gaintables_path, get_plots_path

logger = logging.getLogger()


@ConfigurableStage(name="ionospheric_delay", optional=True)
def ionospheric_delay_stage(
    _upstream_output_,
    _qa_dir_,
    cluster_indexes: Annotated[
        Optional[list[int]],
        Field(
            description=(
                "Array of integers assigning each antenna to a cluster. "
                "If None, all antennas are treated as a single cluster"
            )
        ),
    ] = None,
    block_diagonal: Annotated[
        bool,
        Field(
            description=(
                "If True, solve for all clusters simultaneously assuming a "
                "block-diagonal system. If False, solve for each "
                "cluster sequentially"
            )
        ),
    ] = True,
    niter: Annotated[
        int,
        Field(description="Number of solver iterations."),
    ] = 500,
    tol: Annotated[
        float,
        Field(
            description="""Iteration stops when the fractional change
            in the gain solution is below this tolerance."""
        ),
    ] = 1e-06,
    zernike_limit: Annotated[
        Optional[list[int]],
        Field(
            description="""list of Zernike index limits:
            Generate all Zernikes with n + |m| <= zernike_limit[cluster_id].
            If None, a default is used by the solver."""
        ),
    ] = None,
    timeslice: Annotated[
        Union[float, Literal["auto", "full"]],
        Field(
            description="""Defines time scale over which each gain solution
            is valid. This is used to define time axis of the GainTable.
            This parameter is interpreted as follows,
            float: this is a custom time interval in seconds.
            Input timestamps are grouped by intervals of this duration,
            and said groups are separately averaged to produce
            the output time axis.""",
        ),
    ] = "full",
    plot_table: Annotated[
        bool,
        Field(description="Plot all station Phase vs Frequency"),
    ] = False,
    export_gaintable: Annotated[
        bool,
        Field(description="Export intermediate gain solutions."),
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
     _upstream_output_: dict
        Output from the upstream stage
    _qa_dir_ : str
        Directory path where the diagnostic QA outputs will be written.
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
    zernike_limit : list[int], optional
        list of Zernike index limits:
        Generate all Zernikes with n + |m| <= zernike_limit[cluster_id].
        If None, a default is used by the solver.
    plot_table: bool, optional
        Plot all station Phase vs Frequency (default: False).
    export_gaintable : bool, optional
        If True, the computed gain table is saved to an H5parm file in the
        specified output directory (default: False).

    Returns
    -------
    UpstreamOutput
        The modified upstream_output object, with the vis attribute
        updated to include the applied ionospheric corrections.
    """
    if cluster_indexes is not None:
        cluster_indexes = np.array(cluster_indexes)

    _upstream_output_.add_checkpoint_key("vis")
    prefix = _upstream_output_.ms_prefix
    vis = _upstream_output_.vis
    modelvis = _upstream_output_.modelvis
    vis_chunks = _upstream_output_.chunks
    initialtable = create_gaintable_from_visibility(
        vis, timeslice, "B", skip_default_chunk=True
    )

    gaintable = IonosphericSolver.solve(
        vis,
        modelvis,
        initialtable,
        cluster_indexes,
        block_diagonal,
        niter,
        tol,
        zernike_limit,
    )

    gaintable = gaintable.pipe(with_chunks, vis_chunks)

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=True)
    _upstream_output_["vis"] = vis

    if plot_table:
        path_prefix = get_plots_path(_qa_dir_, f"{prefix}/ionospheric_delay")

        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
            refant=_upstream_output_.refant,
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                gaintable, figure_title="Ionospheric Delay", phase_only=True
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _qa_dir_, f"{prefix}/ionospheric_delay.gaintable.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    return _upstream_output_
