"""Module for generating calibration solutions from visibilities."""

__all__ = [
    "apply_gaintable",
    "solve_bandpass",
]

from typing import Literal

import numpy as np
import xarray
from ska_sdp_func_python.calibration.solvers import solve_gaintable

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.calibration")


def apply_gaintable(
    vis: xarray.Dataset,
    gt: xarray.Dataset,
    inverse: bool = False,
) -> xarray.Dataset:
    """Apply a GainTable to a Visibility.

    This is a temporary local version of
    ska-sdp-func-python.operations.apply_gaintable that avoids a bug in the
    original. Will remove once the ska-sdp-func-python version is fixed. The
    bug is that the function does not transpose the rightmost matrix.

    Note: this is a temporary function and has not been made to robustly
    handle all situations. For instance, it ignores flags and does not
    properly handle sub-bands and partial solution intervals. It also assumes
    that the matrices being applied are Jones matrices and the general
    ska-sdp-datamodels xarrays are used.

    :param vis: Visibility dataset to have gains applied.
    :param gt: GainTable dataset to be applied.
    :param inverse: Apply the inverse. This requires the gain matrices to be
        square. (default=False)
    :return: Input Visibility with gains applied.
    """
    if vis.vis.ndim != gt.gain.ndim - 1:
        raise ValueError("incompatible shapes")
    if vis.vis.shape[-1] != gt.gain.shape[-1] * gt.gain.shape[-1]:
        raise ValueError("incompatible pol axis")
    if inverse and gt.gain.shape[-1] != gt.gain.shape[-2]:
        raise ValueError("gain inversion requires square matrices")

    shape = vis.vis.shape

    # inner pol dim for forward application (and inverse since square)
    npol = gt.gain.shape[-1]

    # need to know which dim has the antennas, so force the structure
    if vis.vis.ndim != 4 or vis.vis.shape[-1] != 4:
        raise ValueError("expecting ska-sdp-datamodels datasets")

    if inverse:
        # use pinv rather than inv to catch singular values
        jones = np.linalg.pinv(gt.gain.data[..., :, :])
    else:
        jones = gt.gain.data

    vis.vis.data = np.einsum(
        "...pi,...ij,...qj->...pq",
        jones[:, vis.antenna1.data],
        vis.vis.data.reshape(shape[0], shape[1], shape[2], npol, npol),
        jones[:, vis.antenna2.data].conj(),
    ).reshape(shape)

    return vis


def solve_bandpass(
    vis: xarray.Dataset,
    modelvis: xarray.Dataset,
    gain_table: xarray.Dataset,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 50,
    phase_only: bool = False,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
    jones_type: Literal["T", "G", "B"] = "T",
    timeslice: float = None,
) -> xarray.Dataset:
    """Determine bandpass calibration Jones matrices.

    The spectral axes need to be consistent. If
    gain_table.frequency==vis.frequency: solve with jones_type="B". If
    gain_table.frequency ~ [mean(vis.frequency)]: solve with jones_type="G".
    Otherwise: raise a ValueError.

    :param vis: Visibility dataset with the unknown corruptions.
    :param modelvis: Visibility model dataset to solve against.
    :param gain_table: GainTable dataset containing bandpass solutions to be
        updated.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0).
    :param niter: Number of solver iterations (defaults to 50).
    :param phase_only: Solve only for the phases.
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").
    :param jones_type: Type of calibration matrix T or G or B.
    :param timeslice: Defines the time scale over which each
        gain solution is valid.

    :return: Updated GainTable dataset.
    """
    logger.debug("solving bandpass")

    # Check inputs
    if refant is not None:
        if refant < 0 or refant >= len(gain_table.antenna):
            raise ValueError(f"invalid refant: {refant}")

    # Check spectral axes
    if gain_table.frequency.equals(vis.frequency):
        jones_type = "B"
    elif len(gain_table.frequency) == 1:
        jones_type = "G"
        if gain_table.frequency.data[0] != np.mean(vis.frequency.data):
            raise ValueError("Single-channel output is at the wrong frequency")
    else:
        raise ValueError("Only supports single-channel or all-channel output")

    # Todo: Set vis flags from a user-defined list of known bad antennas?

    gain_table = solve_gaintable(
        vis=vis,
        modelvis=modelvis,
        gain_table=gain_table,
        solver=solver,
        phase_only=phase_only,
        niter=niter,
        tol=tol,
        crosspol=crosspol,
        normalise_gains=normalise_gains,
        jones_type=jones_type,
        refant=refant,
        timeslice=timeslice,
    )

    # Todo: Check for gain outliers and set flags? RFI or no signal...

    # Todo: Further polarised calibration?

    # Todo: Smoothing? Interpolation over missing channels?
    #  - Careful of sub-band edge effects, but ok if just filling in the gaps.
    #  - Best to leave smoothing until post-processing of all channels?

    # Todo: Return lists of bad antennas and bad channels?

    return gain_table
