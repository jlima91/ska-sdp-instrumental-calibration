# pylint: disable=W1401

import numpy as np
from astropy import constants as const


def generate_rotation_matrices(
    rm: np.ndarray,
    frequency: np.ndarray,
    output_dtype: type = np.complex64,
) -> np.ndarray:
    """
    Generate station rotation matrices based on Rotation Measure (RM) values.

    This function calculates the Faraday rotation angle for each station and
    frequency channel and constructs the corresponding 2x2 rotation matrices.
    These matrices effectively rotate the polarization plane of the incident
    radiation.

    Parameters
    ----------
    rm : numpy.ndarray
        A 1D array containing the Rotation Measure values for each station in
        rad/m^2. Shape: (n_stations,).
    frequency : numpy.ndarray
        A 1D array of frequency channels in Hz. Shape: (n_channels,).
    output_dtype : data-type, optional
        The desired data type of the output rotation matrix. Default is
        ``np.complex64``.

    Returns
    -------
    numpy.ndarray
        A 4D array containing the rotation matrices.
        Shape: (n_stations, n_channels, 2, 2).

    Notes
    -----
    The rotation angle :math:`\phi` is calculated as
    :math:`\phi = \text{RM} \cdot \lambda^2`.

    """  # noqa: W605
    lambda_sq = np.power(
        (const.c.value / frequency), 2  # pylint: disable=E1101
    )

    phi = rm[..., np.newaxis] * lambda_sq

    cos_val = np.cos(phi)
    sin_val = np.sin(phi)

    eye = np.array([[1, 0], [0, 1]])
    A = np.array([[0, -1], [1, 0]])

    rot_array = (
        cos_val[:, :, np.newaxis, np.newaxis] * eye
        + sin_val[:, :, np.newaxis, np.newaxis] * A
    )

    return rot_array.astype(output_dtype)
