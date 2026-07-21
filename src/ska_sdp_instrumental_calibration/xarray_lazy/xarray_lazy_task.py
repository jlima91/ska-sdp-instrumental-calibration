import functools
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar

import dask
import dask.array as da
import xarray as xr

# T represents the type of the first argument
T = TypeVar("T", xr.DataArray, xr.Dataset)
# P represents the rest of the arguments (types and names)
P = ParamSpec("P")


def xarray_lazy_task(
    func: Callable[Concatenate[T, P], Any]
) -> Callable[Concatenate[T, P], xr.DataArray]:
    """
    A replacement for dask.delayed designed specifically for Xarray objects.
    Abuses :py:func:`xarray.map_blocks` to force execution into a native blockwise graph layer,
    guaranteeing perfect graph fusion and avoiding isolated graph recomputations.

    This is useful to wrap operations which are "leaf" tasks in the larger dask graph.
    Such tasks generally produce a side-effects (IO operation, like logs,plots,writing data).
    The result of ``func`` is discarded, and instead we always return an empty xr.DataArray.
    If you wish to preserve the func's return value, then use native methods,
    like :py:func:`xarray.map_blocks`, :py:func:`xarray.apply_ufunc`.
    If you want a function which instead works with pure dask arrays,
    see :py:func:`dask_lazy_task`.

    Example
    -----
    Can be used as a plain decorator, a parameterized decorator, or a direct function wrapper:

    >>> @xarray_lazy_task
    ... def plot_gains(ds, filename):
    ...     ...

    >>> lazy_func = xarray_lazy_task(plot_gains)

    See Also
    --------
    xarray.map_blocks
    xarray.apply_ufunc

    Notes
    -----
    1. Primary Input: The first positional argument (`obj`) must be an Xarray Dataset
       or DataArray. It is automatically consolidated via `.chunk(-1)`.
    2. Flat Positional Arguments: Extra positional arguments (`*args`) must be a flat
       tuple (no nested lists, dicts, or namedtuples).
    3. Xarray-Only for Dask Collections: Any dask-backed collection passed in `*args`
       MUST be wrapped as an Xarray object (Dataset or DataArray) so that `.chunk(-1)`
       can be called on it safely. Pure Dask Arrays or DataFrames are not supported.
    4. No dask in kwargs of ``func``: `xr.map_blocks` does not support dask-backed collections inside
       `**kwargs`. All keyword arguments must be standard, non-dask Python primitives/objects.
    5. Execution Context: When executed on a worker, the function receives fully concrete,
       in-memory Xarray objects for any inputs passed to it.
    """

    @functools.wraps(func)
    def wrapper(obj: T, *args: P.args, **kwargs: P.kwargs) -> xr.DataArray:
        # Consolidate the primary Xarray target object to a single chunk
        collapsed_obj = obj.chunk(-1)

        # Process flat positional arguments (Dask collections are strictly Xarray objects)
        processed_args = tuple(
            arg.chunk(-1) if dask.is_dask_collection(arg) else arg
            for arg in args
        )

        # Setup the default Dask-backed status receipt template
        template = xr.DataArray(
            da.asarray(["pending"]),
            dims=["status"],
            name="_xarray_block_template",
        )

        # The internal executor that runs inside the worker node context
        def _xarray_block_executor_(block, *exec_args, **exec_kwargs):
            func(block, *exec_args, **exec_kwargs)
            return xr.DataArray(["done"], dims=["status"])

        # Hand off directly to the Xarray Blockwise graph engine
        # kwargs are passed through unmodified (no dask collections allowed here)
        return xr.map_blocks(
            _xarray_block_executor_,
            collapsed_obj,
            args=processed_args,
            kwargs=kwargs,
            template=template,
        )

    return wrapper
