import numpy as np


def apply_antenna_gains_to_visibility(
    vis: np.ndarray,
    gains: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    inverse=False,
) -> np.ndarray:
    """
    vis: (time, baselineid, frequency, polarisation)
    gains: (time, antennas, frequency, nrec1, nrec2)
    antenna1: (baselineid)
        Indices of the antenna1 in all baseline pairs
    antenna2: (baselineid)
        Indices of the antenna2 in all baseline pairs
    inverse: bool
        Whether to inverse the gains before applying

    Returns
    -------
    np.ndarray of shape (time, baselineid, frequency, polarisation)
    """
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
