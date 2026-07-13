import functools

import dask
import dask.array as da
import xarray as xr


def xarray_lazy_task(func=None, *, template=None):
    """
    A unified replacement for dask.delayed designed specifically for Xarray objects.
    Abuses `xr.map_blocks` to force execution into a native Blockwise graph layer,
    guaranteeing perfect graph fusion and avoiding isolated graph recomputations.

    Example
    -----
    Can be used as a plain decorator, a parameterized decorator, or a direct function wrapper:

    >>> @xarray_lazy_task
    ... def plot_gains(ds, filename):
    ...     ...

    >>> @xarray_lazy_task(template=custom_output_template)
    ... def compute_metrics(ds):
    ...     ...

    >>> lazy_func = xarray_lazy_task(existing_python_func)

    Notes
    -----
    1. Primary Input: The first positional argument (`obj`) must be an Xarray Dataset
       or DataArray. It is automatically consolidated via `.chunk(-1)`.
    2. Flat Positional Arguments: Extra positional arguments (`*args`) must be a flat
       tuple (no nested lists, dicts, or namedtuples).
    3. Xarray-Only for Dask Collections: Any dask-backed collection passed in `*args`
       MUST be wrapped as an Xarray object (Dataset or DataArray) so that `.chunk(-1)`
       can be called on it safely. Pure Dask Arrays or DataFrames are not supported.
    4. No Dask in kwargs of ``func``: `xr.map_blocks` does not support dask-backed collections inside
       `**kwargs`. All keyword arguments must be standard, non-dask Python primitives/objects.
    5. Execution Context: When executed on a worker, the function receives fully concrete,
       in-memory Xarray objects for any chunked inputs passed to it.
    """
    if func is None:
        return functools.partial(xarray_lazy_task, template=template)

    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        # 1. Consolidate the primary Xarray target object to a single chunk
        collapsed_obj = obj.chunk(-1)

        # 2. Process flat positional arguments (Dask collections are strictly Xarray objects)
        processed_args = tuple(
            arg.chunk(-1) if dask.is_dask_collection(arg) else arg
            for arg in args
        )

        # 3. Setup the default Dask-backed status receipt template
        nonlocal template
        if template is None:
            dummy_dask_arr = da.from_array(["pending"], chunks=(1,))
            template = xr.Dataset({"task_receipt": ("status", dummy_dask_arr)})

        # 4. The internal executor that runs inside the worker node context
        def _block_executor(block, *exec_args, **exec_kwargs):
            result = func(block, *exec_args, **exec_kwargs)

            # Wrap non-xarray primitives cleanly to match the default template schema
            if not isinstance(result, (xr.Dataset, xr.DataArray)):
                return xr.Dataset({"task_receipt": ("status", [str(result)])})
            return result

        # 5. Hand off directly to the Xarray Blockwise graph engine
        # kwargs are passed through unmodified (no dask collections allowed here)
        return xr.map_blocks(
            _block_executor,
            collapsed_obj,
            args=processed_args,
            kwargs=kwargs,
            template=template,
        )

    return wrapper
