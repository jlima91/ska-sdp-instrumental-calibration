"""
Export of MSv4 visibilities to a Processing Set (zarr) or an MSv2.

Both writers keep the computation lazy. They return dask collections, which
the export stage hands to the pipeline task manager, so that the
visibilities are computed once, in the same graph as every other output.
"""

import logging

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility import Visibility

from ..processing_set import (
    get_array_location,
    get_baseline_antenna_indices,
    get_correlated_data,
    get_data_group,
    get_phase_centre,
    field_name_without_xradio_suffix,
    get_field_name,
    get_uvw_for_prediction,
    msv4_time_to_mjd_seconds,
)

logger = logging.getLogger(__name__)

# Zarr format of the Processing Sets written by xradio
PS_ZARR_FORMAT = 3

# Encodings describing the storage layout of the store a variable was read
# from. They do not apply to the chunks of the data being written.
_STORAGE_LAYOUT_ENCODINGS = ("chunks", "preferred_chunks", "shards")

# MSv4 polarizations, in order, to ska-sdp-datamodels polarisation frames
# supported by export_visibility_to_ms, and their receptor frames.
_POLARISATION_FRAMES = {
    ("XX", "XY", "YX", "YY"): ("linear", "linear"),
    ("XX", "YY", "XY", "YX"): ("linearFITS", "linear"),
    ("XX", "YY"): ("linearnp", "linear"),
    ("RR", "RL", "LR", "LL"): ("circular", "circular"),
    ("RR", "LL"): ("circularnp", "circular"),
}


def _is_regular(chunks: tuple[int, ...]) -> bool:
    """
    Whether dask chunks along one dimension can be written to zarr as is,
    i.e. all chunks have the same size, except a smaller last one.
    """
    return all(c == chunks[0] for c in chunks[:-1]) and chunks[-1] <= chunks[0]


def _zarr_writable(ds: xr.Dataset) -> xr.Dataset:
    """
    Shallow copy of a dataset that can be written to a new zarr store.

    - Dimensions with irregular dask chunks in any variable (e.g. the time
      chunks of predicted model visibilities, one per solution interval)
      are merged into a single chunk, as zarr chunks must be uniform.
    - Storage layout encodings of the store the data was read from are
      dropped, so the zarr chunks follow the dask chunks.
    """
    irregular_dims = {
        dim
        for var in ds.variables.values()
        if var.chunks
        for dim, dim_chunks in zip(var.dims, var.chunks)
        if not _is_regular(dim_chunks)
    }
    if irregular_dims:
        logger.info(
            "Merging irregular chunks along %s for zarr export",
            sorted(irregular_dims),
        )
        ds = ds.chunk({dim: -1 for dim in irregular_dims})
    else:
        ds = ds.copy(deep=False)

    for var in ds.variables.values():
        var.encoding = {
            key: value
            for key, value in var.encoding.items()
            if key not in _STORAGE_LAYOUT_ENCODINGS
        }
    return ds


def export_msv4_to_ps(
    ps_path: str, ms_xdt: xr.DataTree, node_name: str
) -> da.Array:
    """
    Write an MSv4 node to a new Processing Set, lazily.

    The Processing Set holds the node, with its child datasets, as its only
    MSv4 node. It can be opened with ``xradio.measurement_set``'s
    ``open_processing_set``, and used again as pipeline input.

    The zarr metadata (groups, attributes, coordinates without dask
    chunks) is written when this function is called. The data is written
    chunk by chunk when the returned task is computed, without gathering it
    in memory.

    Parameters
    ----------
    ps_path
        Path of the Processing Set to write. Overwritten if it exists.
    ms_xdt
        MSv4 node to write.
    node_name
        Name of the MSv4 node in the Processing Set.

    Returns
    -------
        0-d dask array, computing which writes the data.
    """
    ps_xdt = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"type": "processing_set"}),
            node_name: ms_xdt,
        }
    )
    for node in ps_xdt[node_name].subtree:
        node.dataset = _zarr_writable(node.to_dataset(inherit=False))

    write_task = ps_xdt.to_zarr(
        ps_path,
        mode="w",
        zarr_format=PS_ZARR_FORMAT,
        consolidated=True,
        compute=False,
    )
    # Returned as a dask array rather than a Delayed. The task manager
    # persists all registered collections together with graph optimisation,
    # which optimises Delayed and array collections separately. Array keys
    # shared by both (e.g. the corrected visibilities, also exported to
    # MSv2) are then lost, failing the persist. As an array, the write is
    # optimised with the other arrays, in a single graph.
    return da.from_delayed(write_task, shape=(), dtype=object)


def _field_name(ms_xdt: xr.DataTree) -> str:
    """
    MSv2 name of the single field of an MSv4 node, without the
    ``_<FIELD_ID>`` suffix added by xradio.
    """
    return field_name_without_xradio_suffix(get_field_name(ms_xdt))


def _optional_antenna_values(
    antenna_xds: xr.Dataset, name: str, default
) -> np.ndarray:
    """Values of an optional ``antenna_xds`` variable or coordinate."""
    if name in antenna_xds.variables:
        return np.asarray(antenna_xds[name].values)
    return np.asarray(default)


def _msv4_configuration(
    ms_xdt: xr.DataTree, receptor_frame: str
) -> Configuration:
    """
    ska-sdp-datamodels Configuration of the antennas of an MSv4 node, in
    ``antenna_xds`` order, as the MSv2 reader builds it from the ANTENNA
    table.
    """
    antenna_xds = ms_xdt["antenna_xds"].ds
    names = np.asarray(antenna_xds.antenna_name.values)
    nants = names.size
    xyz = antenna_xds.ANTENNA_POSITION.transpose(
        "antenna_name", "cartesian_pos_label"
    ).values

    return Configuration.constructor(
        name="",
        location=get_array_location(ms_xdt),
        names=names,
        xyz=xyz,
        mount=_optional_antenna_values(
            antenna_xds, "mount", np.repeat("alt-az", nants)
        ),
        frame="ITRF",
        receptor_frame=ReceptorFrame(receptor_frame),
        diameter=_optional_antenna_values(
            antenna_xds, "ANTENNA_DISH_DIAMETER", np.zeros(nants)
        ),
        # MSv4 has no antenna OFFSET. It is not written to the MSv2 anyway.
        offset=np.zeros((nants, 3)),
        stations=_optional_antenna_values(antenna_xds, "station_name", names),
    )


def msv4_to_visibility(ms_xdt: xr.DataTree) -> Visibility:
    """
    Lazy ska-sdp-datamodels Visibility of an MSv4 node, for export to MSv2.

    Inverse of the conventions of the MSv2 Visibility reader, so that
    ``export_visibility_to_ms`` writes the data as stored in the MSv4:

    - UVW is negated, as ``export_visibility_to_ms`` negates it on writing.
    - Baselines keep the antenna order of the MSv4, which may have
      antenna1 > antenna2, with unmodified visibilities. Only baselines
      present in the MSv4 are written (no zero-filled autocorrelations or
      missing baselines).
    - Times are converted to seconds since the MJD epoch.

    Visibilities, flags, weights and UVW stay dask arrays, with the chunks
    of the MSv4 node. As for the pre-migration visibility templates,
    ``time_centroid`` is not included. ``_export_visibility_to_ms`` of the
    export stage adds it.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single field and data group.

    Returns
    -------
        Visibility with the data of ``ms_xdt``.

    Raises
    ------
    ValueError
        If the polarizations cannot be written to MSv2, or the integration
        time or channel width is missing.
    """
    correlated_data = get_correlated_data(ms_xdt)
    data_group = get_data_group(ms_xdt)
    data_dims = ("time", "baseline_id", "frequency", "polarization")

    polarization = tuple(
        str(pol) for pol in correlated_data.polarization.values
    )
    if polarization not in _POLARISATION_FRAMES:
        raise ValueError(
            f"Cannot export polarizations {list(polarization)} to MSv2. "
            f"Supported: {[list(p) for p in _POLARISATION_FRAMES]}"
        )
    polarisation_frame, receptor_frame = _POLARISATION_FRAMES[polarization]

    time = ms_xdt.ds.time
    frequency = ms_xdt.ds.frequency
    try:
        integration_time = float(time.attrs["integration_time"]["data"])
        channel_width = float(frequency.attrs["channel_width"]["data"])
    except KeyError as err:
        raise ValueError(
            f"MSv4 node {ms_xdt.name} has no {err} attribute, required for "
            "MSv2 export"
        ) from err

    antenna1, antenna2 = get_baseline_antenna_indices(ms_xdt)
    flags = ms_xdt.ds[data_group["flag"]].transpose(*data_dims)
    weight = ms_xdt.ds[data_group["weight"]].transpose(*data_dims)
    vis_data = correlated_data.transpose(*data_dims).data

    visibility = Visibility.constructor(
        uvw=get_uvw_for_prediction(ms_xdt)
        .transpose("time", "baseline_id", "uvw_label")
        .data,
        baselines=pd.MultiIndex.from_arrays(
            [antenna1, antenna2], names=("antenna1", "antenna2")
        ),
        time=msv4_time_to_mjd_seconds(time.values),
        # Passed explicitly, else the constructor materialises a numpy
        # broadcast of time. Dropped below.
        time_centroid=da.empty(vis_data.shape[:-1], dtype=np.float64),
        frequency=frequency.values,
        channel_bandwidth=np.full(frequency.size, channel_width),
        vis=vis_data,
        flags=flags.data,
        weight=weight.data,
        integration_time=np.full(time.size, integration_time),
        configuration=_msv4_configuration(ms_xdt, receptor_frame),
        phasecentre=get_phase_centre(ms_xdt),
        polarisation_frame=PolarisationFrame(polarisation_frame),
        source=_field_name(ms_xdt),
        low_precision="float64",
    ).drop_vars("time_centroid")

    # The constructor casts weights to float64 and flags to int. Keep the
    # MSv4 dtypes instead, as the pre-migration templates did.
    return visibility.assign(
        weight=(visibility.weight.dims, weight.data, visibility.weight.attrs),
        flags=(visibility.flags.dims, flags.data, visibility.flags.attrs),
    )
