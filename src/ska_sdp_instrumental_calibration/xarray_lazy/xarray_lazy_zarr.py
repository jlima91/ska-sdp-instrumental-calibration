from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from xarray.backends.common import ArrayWriter
from xarray.backends.writers import _datatree_to_zarr as original_xdt_to_zarr
from xarray.backends.writers import (
    _validate_dataset_names,
    dump_to_store,
    get_writable_zarr_store,
)
from xarray.backends.writers import to_zarr as original_xds_to_zarr

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, MutableMapping
    from os import PathLike

    from dask.array import Array as DaskArray
    from xarray import Dataset, DataTree
    from xarray.backends import ZarrStore
    from xarray.core.types import ZarrStoreLike, ZarrWriteModes


__all__ = ["xds_to_zarr", "xdt_to_zarr"]


def _resolve_importable_name(o: type):
    return f"{o.__module__}.{o.__qualname__}"


def xds_to_zarr(
    dataset: Dataset,
    store: ZarrStoreLike | None = None,
    chunk_store: MutableMapping | str | PathLike | None = None,
    mode: ZarrWriteModes | None = None,
    synchronizer=None,
    group: str | None = None,
    encoding: Mapping | None = None,
    *,
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: (
        Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None
    ) = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    zarr_version: int | None = None,
    zarr_format: int | None = None,
    write_empty_chunks: bool | None = None,
    chunkmanager_store_kwargs: dict[str, Any] | None = None,
) -> tuple[DaskArray, ZarrStore]:
    """
    A function to write a dataset to a zarr file.
    Works similar to :py:func:`{original_function}`, except:

    1. The "compute" parameter is removed as the this function is always
       supposed to be lazy. The dataset's datavars must be a dask-based.
    2. Instead of returning a dask.delayed task, this returns a dask array with
       all store-map tasks, and underlying zarr store object.
       Users can use the dask array to create more efficient
       dask graphs.

    Please refer to Example section below.

    See Also
    --------
    {original_function} : Reference for this function's logic.
    xarray.Dataset.to_zarr : For description of all parameters.

    Notes
    -----
    1. The logic in this function must be kept up-to-date with
       the :py:func:`{original_function}`.
    2. To use this function to write a :py:class:`xarray.DataArray` to zarr,
       user should convert the ``DataArray`` to a temporary ``Dataset``
       by calling :py:meth:`xarray.DataArray.to_dataset`,
       and then call this function on the new dataset.
    3. Compared to the :py:func:`{original_function}`, here we do not call
       :py:func:`_finalize_store` on writes and zstore objects.
       As per zarr documentation, it is not required
       to close ZarrStore/ZarrArrays.
       But in case this causes any issues,
       consider closing the zstore post writes are finished,
       in a new dask delayed task.

    Example
    -------
    >>> writes_arr1, _  = {current_function}(dataset1, mode="w")
    >>> writes_arr2, _  = {current_function}(dataset2, mode="w")
    >>> dask.compute(writes_arr1, writes_arr2)
    """

    # validate Dataset keys, DataArray names
    _validate_dataset_names(dataset)

    # Load empty arrays to avoid bug saving zero length dimensions (Issue #5741)
    # TODO: delete when min dask>=2023.12.1
    # https://github.com/dask/dask/pull/10506
    for v in dataset.variables.values():
        if v.size == 0:
            v.load()

    if encoding is None:
        encoding = {}

    zstore = get_writable_zarr_store(
        store,
        chunk_store=chunk_store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        consolidated=consolidated,
        append_dim=append_dim,
        region=region,
        safe_chunks=safe_chunks,
        align_chunks=align_chunks,
        storage_options=storage_options,
        zarr_version=zarr_version,
        zarr_format=zarr_format,
        write_empty_chunks=write_empty_chunks,
    )

    dataset = zstore._validate_and_autodetect_region(dataset)
    zstore._validate_encoding(encoding)

    writer = ArrayWriter()

    dump_to_store(dataset, zstore, writer, encoding=encoding)
    writes = writer.sync(
        compute=False, chunkmanager_store_kwargs=chunkmanager_store_kwargs
    )

    return writes, zstore


xds_to_zarr.__doc__ = xds_to_zarr.__doc__.format(
    original_function=_resolve_importable_name(original_xds_to_zarr),
    current_function=xds_to_zarr.__name__,
)


def xdt_to_zarr(
    dt: DataTree,
    store: ZarrStoreLike,
    mode: ZarrWriteModes = "w-",
    encoding: Mapping[str, Any] | None = None,
    synchronizer=None,
    group: str | None = None,
    write_inherited_coords: bool = False,
    *,
    chunk_store: MutableMapping | str | PathLike | None = None,
    consolidated: bool | None = None,
    append_dim: Hashable | None = None,
    region: (
        Mapping[str, slice | Literal["auto"]] | Literal["auto"] | None
    ) = None,
    safe_chunks: bool = True,
    align_chunks: bool = False,
    storage_options: dict[str, str] | None = None,
    zarr_version: int | None = None,
    zarr_format: int | None = None,
    write_empty_chunks: bool | None = None,
    chunkmanager_store_kwargs: dict[str, Any] | None = None,
) -> tuple[DaskArray, ZarrStore]:
    """
    A function to write a datatree to a zarr file.
    Works similar to :py:func:`{original_function}`, except:

    1. The "compute" parameter is removed as the this function is always
       supposed to be lazy. The datatree must be dask-backed.
    2. Instead of returning a dask.delayed task, this returns a dask array with
       all store-map tasks, and underlying zarr store object.
       Users can use the dask array to create more efficient
       dask graphs.

    Please refer to Example section below.

    See Also
    --------
    {original_function} : Reference for this function's logic.
    xarray.DataTree.to_zarr : For description of all parameters.

    Notes
    -----
    1. The logic in this function must be kept up-to-date with
       the :py:func:`{original_function}`.
    2. Compared to the :py:func:`{original_function}`, here we do not call
       :py:func:`_finalize_store` on writes and root_store objects.
       As per zarr documentation, it is not required
       to close ZarrStore/ZarrArrays.
       But in case this causes any issues,
       consider closing the zstore post writes are finished,
       in a new dask delayed task.

    Example
    -------
    >>> writes_arr1, _  = {current_function}(datatree1, mode="w")
    >>> writes_arr2, _  = {current_function}(datatree2, mode="w")
    >>> dask.compute(writes_arr1, writes_arr2)
    """

    if group is not None:
        raise NotImplementedError(
            "specifying a root group for the tree has not been implemented"
        )

    if append_dim is not None:
        raise NotImplementedError(
            "specifying ``append_dim`` with ``DataTree.to_zarr`` has not been implemented"
        )

    if encoding is None:
        encoding = {}

    # In the future, we may want to expand this check to insure all the provided encoding
    # options are valid. For now, this simply checks that all provided encoding keys are
    # groups in the datatree.
    if set(encoding) - set(dt.groups):
        raise ValueError(
            f"unexpected encoding group name(s) provided: {set(encoding) - set(dt.groups)}"
        )

    root_store = get_writable_zarr_store(
        store,
        chunk_store=chunk_store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        consolidated=consolidated,
        append_dim=append_dim,
        region=region,
        safe_chunks=safe_chunks,
        align_chunks=align_chunks,
        storage_options=storage_options,
        zarr_version=zarr_version,
        zarr_format=zarr_format,
        write_empty_chunks=write_empty_chunks,
    )

    writer = ArrayWriter()

    for rel_path, node in dt.subtree_with_keys:
        at_root = node is dt
        dataset = node.to_dataset(inherit=write_inherited_coords or at_root)
        # Use a relative path for group, because absolute paths are broken
        # with consolidated metadata in zarr 3.1.2 and earlier:
        # https://github.com/zarr-developers/zarr-python/pull/3428
        node_store = (
            root_store if at_root else root_store.get_child_store(rel_path)
        )

        dataset = node_store._validate_and_autodetect_region(dataset)
        node_store._validate_encoding(encoding)

        dump_to_store(
            dataset,
            node_store,
            writer,
            encoding=encoding.get(node.path),
        )

    writes = writer.sync(
        compute=False, chunkmanager_store_kwargs=chunkmanager_store_kwargs
    )

    return writes, root_store


xdt_to_zarr.__doc__ = xdt_to_zarr.__doc__.format(
    original_function=_resolve_importable_name(original_xdt_to_zarr),
    current_function=xdt_to_zarr.__name__,
)
