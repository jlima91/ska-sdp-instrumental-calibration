import functools

import dask
import dask.array as da
import numpy as np
import xarray as xr


def dask_lazy_task(
    func=None, *, dtype=object, chunks=None, **map_blocks_kwargs
):
    """
    A unified replacement for dask.delayed designed specifically for Dask Arrays
    and Xarray DataArrays. Abuses `da.map_blocks` to keep execution inside a native
    Blockwise graph layer, guaranteeing perfect graph fusion.

    Usage
    -----
    Can be used as a plain decorator, a parameterized decorator, or a direct function wrapper:

        @dask_lazy_task
        def export_numpy_data(np_arr, filename):
            ...

        @dask_lazy_task(dtype=np.float64, chunks=(100,))
        def process_math(np_arr):
            ...

        lazy_func = dask_lazy_task(existing_numpy_func)

    Notes
    -----
    1. Primary Input: The first positional argument (`obj`) must be a dask array or a dask-backed
       ``xr.DataArray``.
    1. Only DataArrays Allowed: If any Xarray object other than ``xr.DataArray`` (e.g., ``xr.Dataset``)
       is passed into ``obj`` or ``*args``, a ValueError is raised immediately.
    2. Input Unpacking: Dask collections are automatically collapsed to a single chunk
       using ``.chunk(-1)`` for DataArrays and ``.rechunk(-1)`` for pure Dask Arrays.
    3. Worker Context (NumPy): When running on a worker, all chunked inputs are passed into the
       inner function as fully materialized, concrete NumPy arrays.
    4. Shape Agnostic Receipts: By default, if ``chunks`` is not specified, the wrapper drops
       all dimensions of the input array (``drop_axis``) to return a clean 0D scalar tracking receipt
       containing the function's return string.
    """
    if func is None:
        return functools.partial(
            dask_lazy_task, dtype=dtype, chunks=chunks, **map_blocks_kwargs
        )

    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):

        def _process_input(item):
            # 1. Reject non-DataArray Xarray objects immediately
            if hasattr(item, "__module__") and "xarray" in item.__module__:
                if not isinstance(item, xr.DataArray):
                    raise ValueError(
                        f"Invalid Xarray type '{type(item).__name__}' detected. "
                        "Only xr.DataArray is supported by dask_lazy_task."
                    )

            # 2. Consolidate Dask collections to a single chunk (-1)
            if dask.is_dask_collection(item):
                if isinstance(item, xr.DataArray):
                    return item.chunk(
                        -1
                    ).data  # Consolidate and extract underlying Dask Array
                elif hasattr(item, "rechunk"):
                    return item.rechunk(-1)  # Consolidate pure Dask Array
            return item

        # Process the primary target object
        collapsed_obj = _process_input(obj)

        if not isinstance(collapsed_obj, da.Array):
            raise TypeError(
                "The primary input object must be a Dask Array or a dask-backed xr.DataArray."
            )

        # Process the flat positional arguments array
        processed_args = tuple(_process_input(arg) for arg in args)

        # 3. Handle shape transformations for leaf tasks automatically
        # If no custom shape output is requested, drop all dimensions to create a 0D scalar receipt
        nonlocal chunks
        if chunks is None and "drop_axis" not in map_blocks_kwargs:
            map_blocks_kwargs["drop_axis"] = list(range(collapsed_obj.ndim))
            chunks = ()

        # 4. The internal executor that runs inside the worker node context
        def _block_executor(block, *exec_args, **exec_kwargs):
            # Arguments arrive here as fully concrete NumPy arrays
            result = func(block, *exec_args, **exec_kwargs)

            # Wrap non-array primitives into a NumPy array to satisfy da.map_blocks constraints
            if not isinstance(result, np.ndarray):
                return np.array(str(result), dtype=dtype)
            return result

        # 5. Delegate directly to the Dask Array Blockwise graph generator
        # Hand over processed_args as extra positional parameters to map_blocks
        return da.map_blocks(
            _block_executor,
            collapsed_obj,
            *processed_args,
            dtype=dtype,
            chunks=chunks,
            **map_blocks_kwargs,
        )

    return wrapper
