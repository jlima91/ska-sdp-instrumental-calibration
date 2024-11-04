"""Module for generating calibration solutions from visibilities."""

__all__ = [
    "apply_gaintable",
]

import numpy as np
import xarray

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger(__name__)


def apply_gaintable(
    vis: xarray.Dataset,
    gt: xarray.Dataset,
    inverse: bool = False,
) -> xarray.Dataset:
    """Apply a GainTable to a Visibility.

    This is a temporary local version of
    ska-sdp-func-python.operations.apply_gaintable that avoids a bug in the
    original. Will remove once the ska-sdp-func-python version is fixed. The
    bug is that the function does not transpose the rightmost matrix. This may
    not always be the desired action, for instance if calibration was done
    separately for each polarised visibility type, but is required when the
    solutions are Jones matrices.

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
        raise ValueError("apply_gaintable: incompatible shapes")
    if vis.vis.shape[-1] != gt.gain.shape[-1] * gt.gain.shape[-1]:
        raise ValueError("apply_gaintable: incompatible pol axis")
    if inverse and gt.gain.shape[-1] != gt.gain.shape[-2]:
        raise ValueError(
            "apply_gaintable: gain inversion requires square matrices"
        )

    shape = vis.vis.shape

    # inner pol dim for forward application (and inverse since square)
    npol = gt.gain.shape[-1]

    # need to know which dim has the antennas, so force the structure
    if vis.vis.ndim != 4 or vis.vis.shape[-1] != 4:
        raise ValueError(
            "apply_gaintable: expecting ska-sdp-datamodels xarrays"
        )

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
    gaintable: xarray.Dataset,
) -> xarray.Dataset:
    """Determine bandpass calibration Jones matrices.

    :param vis: Visibility dataset with the unknown corruptions.
    :param modelvis: Visibility model dataset to solve against.
    :param gaintable: GainTable dataset containing initial bandpass solutions.
    :return: Update GainTable dataset.
    """
    logger.info("solving bandpass")

    return gaintable
