from typing import Optional

import xarray as xr
from ska_sdp_func_python.calibration.solvers import solve_gaintable

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    restore_baselines_dim,
)


def _solve_gaintable(
    gainchunk: xr.Dataset,
    vischunk: xr.Dataset,
    modelchunk: xr.Dataset,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
) -> xr.Dataset:
    """Call solve_gaintable.

    Set up to run with function run_solver.

    :param gainchunk: GainTable dataset containing initial solutions.
    :param vischunk: Visibility dataset containing observed data.
    :param modelchunk: Visibility dataset containing model data.
    :param solver: Solver type to use. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0).
    :param niter: Number of solver iterations (defaults to 200).
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").

    :return: Chunked GainTable dataset
    """
    vischunk = restore_baselines_dim(vischunk)
    modelchunk = restore_baselines_dim(modelchunk)

    # Switch to standard variable names and coords for the SDP call
    gainchunk = gainchunk.rename({"frequency_temp": "frequency"})

    gainchunk = solve_gaintable(
        vischunk,
        modelchunk,
        gainchunk,
        True,
        niter,
        tol,
        crosspol,
        normalise_gains,
        solver,
        "G",
        None,
        refant,
    )
    # restore the dimension name back for map_blocks I/O checks
    gainchunk = gainchunk.rename({"frequency": "frequency_temp"})

    return gainchunk


def run_solver(
    vis: xr.Dataset,
    modelvis: xr.Dataset,
    gaintable: xr.Dataset,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: Optional[str] = None,
) -> xr.Dataset:
    """Do the complex gain calibration.

    :param vis: Chunked Visibility dataset containing observed data.
    :param modelvis: Chunked Visibility dataset containing model data.
    :param gaintable: Chunked GainTable dataset containing initial
        solutions.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0). Note that how referencing
        is done depends on the solver.
    :param niter: Number of solver iterations (defaults to 200).
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").

    :return: Chunked GainTable dataset
    """

    # map_blocks won't accept dimensions that differ but have the same name
    # So rename the gain frequency dimension
    gaintable = gaintable.rename({"frequency": "frequency_temp"})

    gaintable = gaintable.map_blocks(
        _solve_gaintable,
        args=[
            vis,
            modelvis,
            solver,
            refant,
            niter,
            tol,
            crosspol,
            normalise_gains,
        ],
        template=gaintable,
    )

    # Undo temporary variable name changes
    gaintable = gaintable.rename({"frequency_temp": "frequency"})
    return gaintable
