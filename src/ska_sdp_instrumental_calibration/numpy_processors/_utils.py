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
    Stacks four ND-array blocks into a 2x2 matrix
    along trailing axes (-2, -1).

    Missing (None) inputs are automatically replaced with zero arrays.
    Supports both Dask and NumPy arrays.
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
