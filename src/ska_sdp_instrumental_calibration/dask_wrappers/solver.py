import logging

import xarray as xr
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.visibility import Visibility

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    restore_baselines_dim,
)
from ska_sdp_instrumental_calibration.processing_tasks.solvers import Solver

logger = logging.getLogger(__name__)


def _run_solver_map_block_(
    vis: Visibility, modelvis: Visibility, gaintable: GainTable, solver: Solver
):
    """
    A map-block compatible wrapper function which internally calls
    `solve_gaintable` function.

    Returns
    -------
    Gaintable
        The gaintable xarray dataset
    """
    # Rename time

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
    A generic function to solve for gaintables, given
    visibility, model visibility and gaintable.

    Parameters
    ----------
    vis: Visibility
        Visibility dataset containing observed data.
    modelvis: Visibility
        Visibility dataset containing model data.
    gaintable: Gaintable
        GainTable dataset containing initial solutions.
    solver: str, default: "gain_substitution"
        Solver type to use. Currently any solver type accepted by
        solve_gaintable.
    refant: int, default: 0
        Reference antenna. Note that how referencing is done
        depends on the solver.
    niter: int, default: 200
        Number of solver iterations.
    phase_only: bool, default: False
        Solve only for the phases.
    tol: float, default: 1e-06
        Iteration stops when the fractional change in the gain solution is
        below this tolerance.
    crosspol: bool, default: False
        Do solutions including cross polarisations.
    normalise_gains: str, default: "mean"
        Normalises the gains.

    Returns
    -------
    GainTable
        A new gaintabel xarray dataset, or the mutated input gaintable
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

    norm_gain = solver.normalise_gains(combined_gaintable.gain)

    return combined_gaintable.assign({"gain": norm_gain})
