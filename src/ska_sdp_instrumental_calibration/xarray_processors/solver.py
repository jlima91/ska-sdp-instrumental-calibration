import logging

import xarray as xr
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.visibility import Visibility

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    restore_baselines_dim,
)

from ..numpy_processors.solvers import Solver

logger = logging.getLogger(__name__)


def _run_solver_map_block_(
    vis: Visibility, modelvis: Visibility, gaintable: GainTable, solver: Solver
) -> GainTable:
    """
    A map-block compatible wrapper function which internally calls
    `solve_gaintable` function.

    Returns
    -------
        A new gaintable containing solutions
    """
    vis = restore_baselines_dim(vis)
    modelvis = restore_baselines_dim(modelvis)

    gain, weight, residual = solver.solve(
        vis.vis.data,
        vis.flags.data,
        vis.weight.data,
        modelvis.vis.data,
        modelvis.flags.data,
        gaintable.gain.data,
        gaintable.weight.data,
        gaintable.residual.data,
        vis.antenna1.data,
        vis.antenna2.data,
    )

    solved_gaintable = gaintable.copy(deep=True)
    solved_gaintable.gain.data = gain
    solved_gaintable.weight.data = weight
    solved_gaintable.residual.data = residual

    return solved_gaintable


def run_solver(
    vis: Visibility,
    modelvis: Visibility,
    gaintable: GainTable,
    solver: Solver,
) -> GainTable:
    """
    A function used for distributing the ``solver.solve()`` function
    call across solution intervals of gaintable, and across the chunks
    of visibility.

    Parameters
    ----------
    vis
        Visibility dataset containing observed data. If its backed by a dask
        array, then it can be chunked in time and frequency axis.
    modelvis
        Visibility dataset containing model data, having similar shape,
        dtype and chunksizes as ``vis``
    gaintable
        GainTable dataset containing initial solutions.
    solver
        An instance of solver, whose ``.solve()``
        method will be called, wrapped in :py:func:`xarray.map_blocks`
        for distributions across dask chunks

    Returns
    -------
        A new gaintable
    """

    vis_chunks_per_solution = {"time": -1}
    gaintable = gaintable.rename(time="solution_time")
    soln_interval_slices = gaintable.soln_interval_slices

    if gaintable.jones_type == "B":
        # solution frequency same as vis frequency
        # Chunking, just to be sure that they match
        gaintable = gaintable.chunk(frequency=vis.chunksizes["frequency"])
    else:  # jones_type == T or G
        assert gaintable.frequency.size == 1, "Gaintable frequency"
        "must either match to visibility frequency, or must be of size 1"
        gaintable = gaintable.rename(frequency="solution_frequency")
        # Need to pass full frequency to process single solution
        vis_chunks_per_solution["frequency"] = -1

    gaintable_across_solutions = []
    for idx, slc in enumerate(soln_interval_slices):
        template_gaintable = gaintable.isel(
            solution_time=[idx]
        )  # Select index but keep dimension
        gaintable_per_solution = xr.map_blocks(
            _run_solver_map_block_,
            vis.isel(time=slc).chunk(vis_chunks_per_solution),
            args=[
                modelvis.isel(time=slc).chunk(vis_chunks_per_solution),
                template_gaintable,
            ],
            kwargs={
                "solver": solver,
            },
            template=template_gaintable,
        )
        gaintable_across_solutions.append(gaintable_per_solution)

    combined_gaintable: GainTable = xr.concat(
        gaintable_across_solutions, dim="solution_time"
    )

    combined_gaintable = combined_gaintable.rename(solution_time="time")

    if "solution_frequency" in combined_gaintable.dims:
        combined_gaintable = combined_gaintable.rename(
            solution_frequency="frequency"
        )

    return combined_gaintable
