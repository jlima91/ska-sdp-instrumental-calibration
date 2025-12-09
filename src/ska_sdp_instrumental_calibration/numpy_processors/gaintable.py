# pylint: disable=W1401

import numpy as np


def apply_antenna_gains_to_visibility(
    vis: np.ndarray,
    gains: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    inverse=False,
) -> np.ndarray:
    """
    Apply antenna gains (Jones matrices) to a visibility array.

    This function modifies the visibility data by applying complex antenna
    gains. Depending on the ``inverse`` parameter, it can be used to simulate
    signal corruption (applying gains directly) or to calibrate the data
    (applying the inverse of the gains).

    Parameters
    ----------
    vis : numpy.ndarray
        The visibility data array.
        Shape: (n_times, n_baselines, n_freqs, n_pols).
        The polarization axis is expected to be flattened (e.g., length 4 for
        XX, XY, YX, YY, corresponding to a 2x2 matrix).
    gains : numpy.ndarray
        The complex antenna gains (Jones matrices).
        Shape: (n_times, n_antennas, n_freqs, n_rec, n_rec).
        Where ``n_rec`` is the number of receptors (typically 2).
    antenna1 : numpy.ndarray
        Indices of the first antenna in each baseline pair.
        Shape: (n_baselines,).
    antenna2 : numpy.ndarray
        Indices of the second antenna in each baseline pair.
        Shape: (n_baselines,).
    inverse : bool, optional
        If True, apply the pseudo-inverse of the gains (calibration).
        If False, apply the gains directly (corruption/simulation).
        Default is False.

    Returns
    -------
    numpy.ndarray
        The modified visibilities.
        Shape: (n_times, n_baselines, n_freqs, n_pols).

    Notes
    -----
    The operation implements the standard radio interferometry measurement
    equation (or its inverse):

    .. math:: V_{out} = G_1 \cdot V_{in} \cdot G_2^H

    where $G_1$ and $G_2$ are the Jones matrices for antenna 1 and antenna 2
    respectively, and $^H$ denotes the Hermitian transpose. To perform this
    multiplication efficiently, the visibility polarization dimension is
    temporarily reshaped into a (2, 2) matrix.
    """  # noqa: W605
    if inverse:
        gains = np.linalg.pinv(gains)

    vis_old_shape = vis.shape
    vis_new_shape = vis.shape[:3] + (2, 2)

    return np.einsum(  # pylint: disable=too-many-function-args
        "tbfpx,tbfxy,tbfqy->tbfpq",
        gains[:, antenna1, ...],
        vis.reshape(vis_new_shape),
        gains[:, antenna2, ...].conj(),
    ).reshape(vis_old_shape)
