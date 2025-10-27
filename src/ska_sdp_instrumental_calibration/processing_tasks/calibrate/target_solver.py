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
    timeslice: float = None,
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
        timeslice,
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
    timeslice: float = None,
    normalise_gains: Optional[str] = None,
) -> xr.Dataset:
    """
    A generic function to solve for gaintables, given
    visibility and model visibility data.

    Parameters
    ----------
    vis: Visibility
        Chunked Visibility dataset containing observed data.
    modelvis: Visibility
        Chunked Visibility dataset containing model data.
    gaintable: Gaintable, optional
        Optional chunked GainTable dataset containing initial solutions.
    solver: str, default: "gain_substitution"
        Solver type to use. Currently any solver type accepted by
        solve_gaintable.
    refant: int, default: 0
        Reference antenna. Note that how referencing is done
        depends on the solver.
    niter: int, default: 200
        Number of solver iterations.
    tol: float, default: 1e-06
        Iteration stops when the fractional change in the gain solution is
        below this tolerance.
    crosspol: bool, default: False
        Do solutions including cross polarisations.
    normalise_gains: str, default: "mean"
        Normalises the gains.
    timeslice: float, optional
        Defines the time scale over which each gain solution is valid.

    Returns
    -------
    GainTable
        A new gaintabel xarray dataset, or the mutated input gaintable
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
            timeslice,
        ],
        template=gaintable,
    )

    # Undo temporary variable name changes
    gaintable = gaintable.rename({"frequency_temp": "frequency"})
    return gaintable
