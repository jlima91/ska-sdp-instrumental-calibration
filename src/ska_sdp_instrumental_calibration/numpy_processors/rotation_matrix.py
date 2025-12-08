import numpy as np
from astropy import constants as const


def generate_rotation_matrices(
    rm: np.ndarray,
    frequency: np.ndarray,
    output_dtype: type = np.complex64,
) -> np.ndarray:
    """Generate station rotation matrix from RM values.

    :param rm: 1D array of rotation measure values [nstation].
    :param frequency: 1D array of frequency values [nfrequency].
    :param output_dtype: output dtype of rotation matrix

    :return: 4D array of rotation matrix: [nstation, nfrequency, 2, 2].
    """
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
