import functools
from typing import Any, Callable, Concatenate, ParamSpec

import dask
import dask.array as da

# P represents the rest of the arguments (types and names)
P = ParamSpec("P")


def dask_lazy_task(
    func: Callable[Concatenate[da.Array, P], Any]
) -> Callable[Concatenate[da.Array, P], da.Array]:
    """
    A replacement for ``dask.delayed`` designed specifically for dask arrays.
    Abuses :py:func:`dask.array.map_blocks` to keep execution inside a native
    blockwise graph layer, guaranteeing perfect graph fusion.

    This is useful to wrap operations which are "leaf" tasks in the larger dask graph.
    Such tasks generally produce a side-effects (IO operation, like logs,plots,writing data).
    The result of ``func`` is discarded, and instead we always return an empty np.ndarray.
    If you wish to preserve the func's return value, then use native methods,
    like :py:func:`dask.array.map_blocks`, :py:func:`dask.array.gufunc.apply_gufunc`.
    If you want a function which instead works with dask-backed xarray objects,
    see :py:func:`xarray_lazy_task`.

    Usage
    -----
    Can be used as a plain decorator, a parameterized decorator, or a direct function wrapper:

    >>> @dask_lazy_task
    ... def export_numpy_data(np_arr, filename):
    ...     ...

    >>> lazy_func = dask_lazy_task(export_numpy_data)

    See Also
    --------
    dask.array.map_blocks
    dask.array.gufunc.apply_gufunc

    Notes
    -----
    1. Primary Input: The first positional argument (`obj`) must be a dask array.
    2. Flat Positional Arguments: Extra positional arguments (`*args`) must be a flat
       tuple (no nested lists, dicts, or namedtuples).
    3. Input rechunking: All dask collections are automatically collapsed to a single chunk
    4. No dask in kwargs of ``func``: `xr.map_blocks` does not support dask-backed collections inside
       `**kwargs`. All keyword arguments must be standard, non-dask Python primitives/objects.
    5. Execution Context: When executed on a worker, all inputs are passed into the
       inner function as fully materialized, concrete NumPy arrays.
    """

    @functools.wraps(func)
    def wrapper(obj: da.Array, *args: P.args, **kwargs: P.kwargs) -> da.Array:
        # Process the primary target object
        # Here we assume that item is a dask.array-like
        # which implements "rechunk()" function
        collapsed_obj = obj.rechunk(-1)

        # Process the flat positional arguments array
        processed_args = tuple(
            arg.rechunk(-1) if dask.is_dask_collection(arg) else arg
            for arg in args
        )

        # The internal executor that runs inside the worker node context
        def _dask_block_executor_(block, *exec_args, **exec_kwargs):
            # discard function result
            func(block, *exec_args, **exec_kwargs)

        # Delegate directly to the Dask Array Blockwise graph generator
        # Hand over processed_args as extra positional parameters to map_blocks
        return da.map_blocks(
            _dask_block_executor_,
            collapsed_obj,
            *processed_args,
            meta=da.asarray(None),
            **kwargs,
        )

    return wrapper
