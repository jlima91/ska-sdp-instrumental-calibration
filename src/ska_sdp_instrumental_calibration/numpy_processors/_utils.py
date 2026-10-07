from typing import TypeVar

import dask.array as da
import numpy as np

T_AnyArray = TypeVar("AnyArray", np.ndarray, da.Array)


def stack_2x2(
    xx: T_AnyArray = None,
    xy: T_AnyArray = None,
    yx: T_AnyArray = None,
    yy: T_AnyArray = None,
) -> T_AnyArray:
    """
    Stacks four ND-array blocks into a 2x2 matrix along trailing axes (-2, -1).

    Missing (None) inputs are automatically replaced with zero arrays.
    Supports both Dask and NumPy arrays.

    Parameters
    ----------
    xx
        Top-left block, shape (...).
    xy
        Top-right block, shape (...).
    yx
        Bottom-left block, shape (...).
    yy
        Bottom-right block, shape (...).

    Returns
    -------
        Array of shape (..., 2, 2) where the trailing two axes form the 2x2
        matrix ``[[xx, xy], [yx, yy]]``.

    Raises
    ------
    ValueError
        If all four inputs are None.

    Examples
    --------
    >>> import numpy as np
    >>> xx = np.ones((3, 4))
    >>> yy = np.full((3, 4), 2.0)
    >>> result = stack_2x2(xx=xx, yy=yy)
    >>> result.shape
    (3, 4, 2, 2)
    >>> result[0, 0]
    array([[1., 0.],
           [0., 2.]])
    """
    inputs = [xx, xy, yx, yy]

    ref = next((x for x in inputs if x is not None), None)
    if ref is None:
        raise ValueError("At least one input array must be provided.")

    xp = da if isinstance(ref, da.Array) else np

    filled = [x if x is not None else xp.zeros_like(ref) for x in inputs]
    xx_f, xy_f, yx_f, yy_f = filled

    row0 = xp.stack([xx_f, xy_f], axis=-1)  # [XX, XY]
    row1 = xp.stack([yx_f, yy_f], axis=-1)  # [YX, YY]

    return xp.stack([row0, row1], axis=-2)


# Polarization size -> receptors per antenna
NPOL_TO_NREC = {1: 1, 4: 2}


def pol_to_jones(x: T_AnyArray) -> T_AnyArray:
    """
    Convert polarization into (2x2 or 1x1) Jones matrices.

    [..., npol] -> [..., nrec, nrec]
    """
    nrec = NPOL_TO_NREC[x.shape[-1]]
    return x.reshape(*x.shape[:-1], nrec, nrec)


def jones_to_pol(x: T_AnyArray) -> T_AnyArray:
    """
    Flatten Jones matrices into polarization.

    [..., nrec, nrec] -> [..., npol]
    """
    return x.reshape(*x.shape[:-2], -1)
