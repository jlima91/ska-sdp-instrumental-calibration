"""
Polarization helpers for the flat ``polarization`` axis of
GainCalibrationSetXds.
"""

from typing import TypeVar

import dask.array as da
import numpy as np

T_AnyArray = TypeVar("T_AnyArray", np.ndarray, da.Array)

# Polarization size -> receptors per antenna
NPOL_TO_NREC = {1: 1, 4: 2}

# Flattened 2x2 Jones labels, in row-major (receptor1, receptor2) order
LINEAR_POLS = ("XX", "XY", "YX", "YY")
CIRCULAR_POLS = ("RR", "RL", "LR", "LL")


def npol_to_nrec(npol: int) -> int:
    """
    Number of receptors per antenna for a polarization axis size.

    Parameters
    ----------
    npol
        Size of the polarization axis.

    Returns
    -------
        Number of receptors, 1 for scalar gains or 2 for 2x2 Jones.

    Raises
    ------
    ValueError
        If ``npol`` is not 1 or 4.
    """
    try:
        return NPOL_TO_NREC[npol]
    except KeyError:
        raise ValueError(
            f"Unsupported number of polarizations: {npol}. "
            f"Expected one of {sorted(NPOL_TO_NREC)}"
        ) from None


def pol_index(rec1: int, rec2: int, nrec: int = 2) -> int:
    """
    Index of a Jones element on the flat polarization axis.

    The polarization axis holds the ``(nrec, nrec)`` Jones matrix
    flattened in row-major order, as done by :func:`jones_to_pol`.

    Parameters
    ----------
    rec1
        Receptor index of the first antenna (Jones row).
    rec2
        Receptor index of the second antenna (Jones column).
    nrec
        Number of receptors per antenna.

    Returns
    -------
        Index along the polarization axis, e.g. 3 for YY when nrec is 2.

    Raises
    ------
    ValueError
        If ``rec1`` or ``rec2`` is not in ``range(nrec)``.
    """
    if not (0 <= rec1 < nrec and 0 <= rec2 < nrec):
        raise ValueError(
            f"Receptor indices ({rec1}, {rec2}) out of range for "
            f"nrec={nrec}"
        )
    return rec1 * nrec + rec2


def pol_to_jones(x: T_AnyArray) -> T_AnyArray:
    """
    Convert polarization into (2x2 or 1x1) Jones matrices.

    [..., npol] -> [..., nrec, nrec]

    Parameters
    ----------
    x
        Array with the polarization axis last.

    Returns
    -------
        Array with the trailing axis reshaped to ``(nrec, nrec)``.

    Raises
    ------
    ValueError
        If the trailing axis size is not 1 or 4.
    """
    nrec = npol_to_nrec(x.shape[-1])
    return x.reshape(*x.shape[:-1], nrec, nrec)


def jones_to_pol(x: T_AnyArray) -> T_AnyArray:
    """
    Flatten Jones matrices into polarization.

    [..., nrec, nrec] -> [..., npol]

    Parameters
    ----------
    x
        Array with the two trailing axes forming Jones matrices.

    Returns
    -------
        Array with the trailing two axes flattened.
    """
    return x.reshape(*x.shape[:-2], -1)
