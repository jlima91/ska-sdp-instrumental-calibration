import logging

import numpy as np
import xarray as xr
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.visibility import Visibility

from ..numpy_processors.solvers import Solver

logger = logging.getLogger(__name__)


def _run_solver_ufunc(
    vis_vis: np.ndarray,
    vis_flags: np.ndarray,
    vis_weight: np.ndarray,
    model_vis: np.ndarray,
    model_flags: np.ndarray,
    gain_gain: np.ndarray,
    gain_weight: np.ndarray,
    gain_residual: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    solver: Solver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A bridge function between xarray.apply_ufunc and solver.solve.

    Returns
    -------
        Gain, weight and residual arrays returned by the solver.
    """
    return solver.solve(
        vis_vis,
        vis_flags,
        vis_weight,
        model_vis,
        model_flags,
        gain_gain,
        gain_weight,
        gain_residual,
        antenna1,
        antenna2,
    )


def _run_solver_ufunc_with_broadcast_frequency(
    vis_vis: np.ndarray,
    vis_flags: np.ndarray,
    vis_weight: np.ndarray,
    model_vis: np.ndarray,
    model_flags: np.ndarray,
    gain_gain: np.ndarray,
    gain_weight: np.ndarray,
    gain_residual: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    solver: Solver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solver wrapper for bandpass terms where frequency stays broadcast.

    xarray passes broadcast dimensions ahead of core dimensions. For Jones
    type B we keep frequency outside the core dims so dask can still
    distribute work over visibility frequency chunks.
    """
    gain, weight, residual = _run_solver_ufunc(
        vis_vis=np.transpose(vis_vis, (1, 2, 0, 3)),
        vis_flags=np.transpose(vis_flags, (1, 2, 0, 3)),
        vis_weight=np.transpose(vis_weight, (1, 2, 0, 3)),
        model_vis=np.transpose(model_vis, (1, 2, 0, 3)),
        model_flags=np.transpose(model_flags, (1, 2, 0, 3)),
        gain_gain=np.transpose(gain_gain, (1, 2, 0, 3, 4)),
        gain_weight=np.transpose(gain_weight, (1, 2, 0, 3, 4)),
        gain_residual=np.transpose(gain_residual, (1, 0, 2, 3)),
        antenna1=antenna1,
        antenna2=antenna2,
        solver=solver,
    )

    return (
        np.transpose(gain, (2, 0, 1, 3, 4)),
        np.transpose(weight, (2, 0, 1, 3, 4)),
        np.transpose(residual, (1, 0, 2, 3)),
    )


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
        method will be called, wrapped in :py:func:`xarray.apply_ufunc`
        for distributions across dask chunks

    Returns
    -------
        A new gaintable
    """

    vis_chunks_per_solution = {"time": -1}
    gaintable = gaintable.rename(time="solution_time")
    soln_interval_slices = gaintable.soln_interval_slices
    output_dtypes = [
        gaintable.gain.dtype,
        gaintable.weight.dtype,
        gaintable.residual.dtype,
    ]

    if gaintable.jones_type == "B":
        # solution frequency same as vis frequency
        # Chunking, just to be sure that they match
        gaintable = gaintable.chunk(frequency=vis.chunksizes["frequency"])
        solver_ufunc = _run_solver_ufunc_with_broadcast_frequency
        vis_core_dims = ["time", "baselineid", "polarisation"]
        gain_core_dims = [
            "solution_time",
            "antenna",
            "receptor1",
            "receptor2",
        ]
        residual_core_dims = [
            "solution_time",
            "receptor1",
            "receptor2",
        ]
    else:  # jones_type == T or G
        assert gaintable.frequency.size == 1, "Gaintable frequency"
        "must either match to visibility frequency, or must be of size 1"
        gaintable = gaintable.rename(frequency="solution_frequency")
        # Need to pass full frequency to process single solution
        vis_chunks_per_solution["frequency"] = -1
        solver_ufunc = _run_solver_ufunc
        vis_core_dims = ["time", "baselineid", "frequency", "polarisation"]
        gain_core_dims = [
            "solution_time",
            "antenna",
            "solution_frequency",
            "receptor1",
            "receptor2",
        ]
        residual_core_dims = [
            "solution_time",
            "solution_frequency",
            "receptor1",
            "receptor2",
        ]

    gaintable_across_solutions = []
    for idx, slc in enumerate(soln_interval_slices):
        vis_per_solution = vis.isel(time=slc).chunk(vis_chunks_per_solution)
        modelvis_per_solution = modelvis.isel(time=slc).chunk(
            vis_chunks_per_solution
        )
        template_gaintable = gaintable.isel(
            solution_time=[idx]
        )  # Select index but keep dimension
        gain, weight, residual = xr.apply_ufunc(
            solver_ufunc,
            vis_per_solution.vis,
            vis_per_solution.flags,
            vis_per_solution.weight,
            modelvis_per_solution.vis,
            modelvis_per_solution.flags,
            template_gaintable.gain,
            template_gaintable.weight,
            template_gaintable.residual,
            input_core_dims=[
                vis_core_dims,
                vis_core_dims,
                vis_core_dims,
                vis_core_dims,
                vis_core_dims,
                gain_core_dims,
                gain_core_dims,
                residual_core_dims,
            ],
            output_core_dims=[
                gain_core_dims,
                gain_core_dims,
                residual_core_dims,
            ],
            dask="parallelized",
            output_dtypes=output_dtypes,
            kwargs={
                "antenna1": vis.antenna1.data,
                "antenna2": vis.antenna2.data,
                "solver": solver,
            },
        )
        gaintable_per_solution = template_gaintable.assign(
            gain=gain.transpose(*template_gaintable.gain.dims),
            weight=weight.transpose(*template_gaintable.weight.dims),
            residual=residual.transpose(*template_gaintable.residual.dims),
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
