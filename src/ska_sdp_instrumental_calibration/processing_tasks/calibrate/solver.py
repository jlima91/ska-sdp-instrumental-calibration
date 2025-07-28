import logging
from typing import Literal, Optional

import numpy as np
import xarray as xr
from ska_sdp_func_python.calibration.solvers import solve_gaintable

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    restore_baselines_dim,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    create_bandpass_table,
)

logger = logging.getLogger()


def _solve_gaintable(
    gain,
    vis,
    modelvis,
    phase,
    niter,
    tol,
    crosspol,
    normalise,
    solver,
    jones,
    timeslice,
    refant,
):
    """
    A map-block compatible wrapper function which internally calls
    `solve_gaintable` function.

    Returns
    -------
    Gaintable
        The gaintable xarray dataset
    """
    gain = gain.rename({"soln_time": "time"})

    return solve_gaintable(
        vis,
        modelvis,
        gain,
        phase,
        niter,
        tol,
        crosspol,
        normalise,
        solver,
        jones,
        timeslice,
        refant,
    ).rename({"time": "soln_time"})


def run_solver(
    vis: xr.Dataset,
    modelvis: xr.Dataset,
    gaintable: Optional[xr.Dataset] = None,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    phase_only: bool = False,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
    jones_type: Literal["T", "G", "B"] = "T",
    timeslice: float = None,
):
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
    phase_only: bool, default: False
        Solve only for the phases.
    tol: float, default: 1e-06
        Iteration stops when the fractional change in the gain solution is
        below this tolerance.
    crosspol: bool, default: False
        Do solutions including cross polarisations.
    normalise_gains: str, default: "mean"
        Normalises the gains.
    jones_type: Literal["T", "G", "B"], default: "T"
        Type of calibration matrix T or G or B.
    timeslice: float, optional
        Defines the time scale over which each gain solution is valid.

    Returns
    -------
    GainTable
        A new gaintabel xarray dataset, or the mutated input gaintable
    """

    if gaintable is None:
        fchunk = vis.chunks["frequency"][0]
        if fchunk <= 0:
            logger.warning("vis dataset does not appear to be chunked")
            fchunk = len(vis.frequency)
        gaintable = create_bandpass_table(vis).chunk({"frequency": fchunk})

    if len(gaintable.time) != 1:
        raise ValueError(
            "Error setting up gaintable. Size of 'time' dimension is not 1."
        )

    if refant < 0 or refant >= len(gaintable.antenna):
        raise ValueError(f"Invalid refant: {refant}")

    # Check spectral axes
    if gaintable.frequency.equals(vis.frequency):
        jones_type = "B"
    elif len(gaintable.frequency) == 1:
        jones_type = "G"
        if gaintable.frequency[0] != np.mean(vis.frequency):
            raise ValueError("Single-channel output is at the wrong frequency")
    else:
        raise ValueError("Only supports single-channel or all-channel output")

    # map_blocks won't accept dimensions that differ but have the same name
    # So rename the gain time dimension (and coordinate)
    # Switch to standard variable names and coords for the SDP call
    gaintable = gaintable.rename({"time": "soln_time"})

    restored_vis = restore_baselines_dim(vis)
    restored_modelvis = restore_baselines_dim(modelvis)

    gaintable = gaintable.map_blocks(
        _solve_gaintable,
        args=[
            restored_vis,
            restored_modelvis,
            phase_only,
            niter,
            tol,
            crosspol,
            normalise_gains,
            solver,
            jones_type,
            timeslice,
            refant,
        ],
        template=gaintable,
    )
    # Change the time dimension name back for map_blocks I/O checks
    gaintable = gaintable.rename({"soln_time": "time"})

    return gaintable
