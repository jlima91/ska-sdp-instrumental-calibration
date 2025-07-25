import logging
from typing import Literal, Optional

import numpy as np
import xarray as xr
from ska_sdp_func_python.calibration.solvers import solve_gaintable

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
) -> xr.Dataset:
    """Do the bandpass calibration.

    :param vis: Chunked Visibility dataset containing observed data.
    :param modelvis: Chunked Visibility dataset containing model data.
    :param gaintable: Optional chunked GainTable dataset containing initial
        solutions.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0). Note that how referencing
        is done depends on the solver.
    :param niter: Number of solver iterations (defaults to 200).
    :param phase_only: Solve only for the phases.
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").
    :param jones_type: Type of calibration matrix T or G or B.
    :param timeslice: Defines the time scale over which each
        gain solution is valid.

    :return: Chunked GainTable dataset
    """

    if gaintable is None:
        fchunk = vis.chunks["frequency"][0]
        if fchunk <= 0:
            logger.warning("vis dataset does not appear to be chunked")
            fchunk = len(vis.frequency)
        gaintable = create_bandpass_table(vis).chunk({"frequency": fchunk})

    if len(gaintable.time) != 1:
        raise ValueError("error setting up gaintable")

    if refant is not None:
        if refant < 0 or refant >= len(gaintable.antenna):
            raise ValueError(f"invalid refant: {refant}")

    # Call solver
    logger.debug("solving bandpass")

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

    gaintable = gaintable.map_blocks(
        _solve_gaintable,
        args=[
            vis,
            modelvis,
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
