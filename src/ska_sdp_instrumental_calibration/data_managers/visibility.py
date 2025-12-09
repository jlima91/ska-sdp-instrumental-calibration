import logging
import os
import pickle
from copy import deepcopy
from typing import List, Optional

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas
import xarray as xr
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from casacore.tables import table, taql
from dask.delayed import Delayed
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ..xarray_processors import simplify_baselines_dim, with_chunks

logger = logging.getLogger(__name__)

ATTRS_FILE_NAME = "attrs.pickle"
VIS_FILE_NAME = "vis.zarr"
BASELINE_FILE_NAME = "baselines.pickle"


def create_template_vis_from_ms(
    msname: str,
    ack: bool = False,
    datacolumn: str = "DATA",
    field_ids: List[int] = None,
    data_desc_ids: List[int] = None,
) -> List[Visibility]:
    """
    Create empty "template" visibility objects from a Measurement Set.

    This function inspects the provided Measurement Set (MS) to determine
    the shapes, types, and metadata required to create Visibility objects.
    It returns a list of these objects where the data arrays (vis, flags,
    weights, uvw) are initialized as empty Dask arrays. These templates
    can be populated later.

    Parameters
    ----------
    msname : str
        The file path to the Measurement Set.
    ack : bool, optional
        If True, print an acknowledgement message when opening the table.
        Default is False.
    datacolumn : str, optional
        The name of the column in the MS to use for determining the data
        type of the visibility data. Default is "DATA".
    field_ids : list[int], optional
        A list of field IDs to process. If None, defaults to [0].
    data_desc_ids : list[int], optional
        A list of data description IDs to process. If None, defaults to [0].

    Returns
    -------
    list[Visibility]
        A list of Visibility objects corresponding to the selected field
        and data description IDs. The data arrays within are empty Dask
        arrays.

    Raises
    ------
    ValueError
        If the selection for a specific Field ID or Data Description ID
        yields no rows in the MS.
    KeyError
        If the polarization configuration in the MS is not recognized.
    """
    field_ids = field_ids or [0]
    data_desc_ids = data_desc_ids or [0]

    vis_list = []

    with table(msname, ack=ack, readonly=True) as tab:
        for field in field_ids:
            with tab.query(f"FIELD_ID=={field}", style="") as ftab:
                if ftab.nrows() <= 0:
                    raise ValueError(f"Empty selection for FIELD_ID={field}")

                for dd in data_desc_ids:
                    with table(
                        f"{msname}/DATA_DESCRIPTION", ack=ack, readonly=True
                    ) as ddtab:
                        spwid = ddtab.getcol("SPECTRAL_WINDOW_ID")[dd]
                        polid = ddtab.getcol("POLARIZATION_ID")[dd]

                    meta = {"MSV2": {"FIELD_ID": field, "DATA_DESC_ID": dd}}

                    with ftab.query(f"DATA_DESC_ID=={dd}", style="") as ms:
                        if ms.nrows() <= 0:
                            raise ValueError(
                                f"Empty selection for FIELD_ID= {field} "
                                f"and DATA_DESC_ID={dd}"
                            )

                        otime = ms.getcol("TIME")
                        antenna1 = ms.getcol("ANTENNA1")
                        antenna2 = ms.getcol("ANTENNA2")
                        integration_time = ms.getcol("INTERVAL")

                        vis_dtype = ms.getcol(datacolumn, nrow=1).dtype
                        flags_dtype = ms.getcol("FLAG", nrow=1).dtype
                        weight_dtype = ms.getcol("WEIGHT", nrow=1).dtype
                        uvw_dtype = ms.getcol("UVW", nrow=1).dtype

                    time = otime - integration_time / 2.0
                    start_time = np.min(time) / 86400.0
                    end_time = np.max(time) / 86400.0

                    logger.debug(
                        "create_visibility_from_ms: Observation from %s to %s",
                        Time(start_time, format="mjd").iso,
                        Time(end_time, format="mjd").iso,
                    )

                    with table(
                        f"{msname}/SPECTRAL_WINDOW", ack=ack, readonly=True
                    ) as spwtab:
                        cfrequency = np.array(
                            spwtab.getcol("CHAN_FREQ")[spwid]
                        )
                        cchannel_bandwidth = np.array(
                            spwtab.getcol("CHAN_WIDTH")[spwid]
                        )

                    nchan = cfrequency.shape[0]

                    # Get polarisation info
                    with table(
                        f"{msname}/POLARIZATION", ack=ack, readonly=True
                    ) as poltab:
                        corr_type = poltab.getcol("CORR_TYPE")[polid]

                    # These correspond to the CASA Stokes enumerations
                    if np.array_equal(corr_type, [1, 2, 3, 4]):
                        polarisation_frame = PolarisationFrame("stokesIQUV")
                        npol = 4
                    elif np.array_equal(corr_type, [1, 2]):
                        polarisation_frame = PolarisationFrame("stokesIQ")
                        npol = 2
                    elif np.array_equal(corr_type, [1, 4]):
                        polarisation_frame = PolarisationFrame("stokesIV")
                        npol = 2
                    elif np.array_equal(corr_type, [5, 6, 7, 8]):
                        polarisation_frame = PolarisationFrame("circular")
                        npol = 4
                    elif np.array_equal(corr_type, [5, 8]):
                        polarisation_frame = PolarisationFrame("circularnp")
                        npol = 2
                    elif np.array_equal(corr_type, [9, 10, 11, 12]):
                        polarisation_frame = PolarisationFrame("linear")
                        npol = 4
                    elif np.array_equal(corr_type, [9, 12, 10, 11]):
                        polarisation_frame = PolarisationFrame("linearFITS")
                        npol = 4
                    elif np.array_equal(corr_type, [9, 12]):
                        polarisation_frame = PolarisationFrame("linearnp")
                        npol = 2
                    elif np.array_equal(corr_type, [9]) or np.array_equal(
                        corr_type, [1]
                    ):
                        npol = 1
                        polarisation_frame = PolarisationFrame("stokesI")
                    else:
                        raise KeyError(
                            f"Polarisation not understood: {str(corr_type)}"
                        )

                    # Get configuration
                    with table(
                        f"{msname}/ANTENNA", ack=ack, readonly=True
                    ) as anttab:
                        names = np.array(anttab.getcol("NAME"))

                        # pylint: disable=cell-var-from-loop
                        ant_map = []
                        actual = 0
                        # This assumes that the names are actually filled in!
                        for name in names:
                            if name != "":
                                ant_map.append(actual)
                                actual += 1
                            else:
                                ant_map.append(-1)

                        if actual == 0:
                            ant_map = list(range(len(names)))
                            names = np.repeat("No name", len(names))

                        mount = np.array(anttab.getcol("MOUNT"))[names != ""]
                        # logger.info("mount is: %s" % (mount))
                        diameter = np.array(anttab.getcol("DISH_DIAMETER"))[
                            names != ""
                        ]
                        xyz = np.array(anttab.getcol("POSITION"))[names != ""]
                        offset = np.array(anttab.getcol("OFFSET"))[names != ""]
                        stations = np.array(anttab.getcol("STATION"))[
                            names != ""
                        ]
                        names = np.array(anttab.getcol("NAME"))[names != ""]
                        nants = len(names)

                    antenna1 = list(map(lambda i: ant_map[i], antenna1))
                    antenna2 = list(map(lambda i: ant_map[i], antenna2))

                    baselines = pandas.MultiIndex.from_arrays(
                        np.triu_indices(nants, k=0),
                        names=("antenna1", "antenna2"),
                    )
                    nbaselines = len(baselines)

                    location = EarthLocation(
                        x=Quantity(xyz[0][0], "m"),
                        y=Quantity(xyz[0][1], "m"),
                        z=Quantity(xyz[0][2], "m"),
                    )

                    configuration = Configuration.constructor(
                        name="",
                        location=location,
                        names=names,
                        xyz=xyz,
                        mount=mount,
                        frame="ITRF",
                        receptor_frame=ReceptorFrame("linear"),
                        diameter=diameter,
                        offset=offset,
                        stations=stations,
                    )
                    # Get phasecentres
                    with table(
                        f"{msname}/FIELD", ack=ack, readonly=True
                    ) as fieldtab:
                        pc = fieldtab.getcol("PHASE_DIR")[field, 0, :]
                        source = fieldtab.getcol("NAME")[field]

                    phasecentre = SkyCoord(
                        ra=pc[0] * u.rad,
                        dec=pc[1] * u.rad,
                        frame="icrs",
                        equinox="J2000",
                    )

                    time_index_row = np.zeros_like(time, dtype="int")
                    time_last = time[0]
                    time_index = 0
                    for row, _ in enumerate(time):
                        if time[row] > time_last + 0.5 * integration_time[row]:
                            assert (
                                time[row] > time_last
                            ), "MS is not time-sorted - cannot convert"
                            time_index += 1
                            time_last = time[row]
                        time_index_row[row] = time_index

                    ntimes = time_index + 1

                    assert ntimes == len(
                        np.unique(time_index_row)
                    ), "Error in finding data times"

                    bv_vis = da.empty(
                        [ntimes, nbaselines, nchan, npol], dtype=vis_dtype
                    )
                    bv_flags = da.empty(
                        [ntimes, nbaselines, nchan, npol], dtype=flags_dtype
                    )
                    bv_weight = da.empty(
                        [ntimes, nbaselines, nchan, npol], dtype=weight_dtype
                    )
                    bv_uvw = da.empty([ntimes, nbaselines, 3], dtype=uvw_dtype)

                    bv_times = np.zeros([ntimes])
                    bv_integration_time = np.zeros([ntimes])

                    for row, _ in enumerate(time):
                        time_index = time_index_row[row]
                        bv_times[time_index] = time[row]
                        bv_integration_time[time_index] = integration_time[row]

                    vis_template = Visibility.constructor(
                        uvw=bv_uvw,
                        baselines=baselines,
                        time=bv_times,
                        frequency=cfrequency,
                        channel_bandwidth=cchannel_bandwidth,
                        vis=bv_vis,
                        flags=bv_flags,
                        weight=bv_weight,
                        integration_time=bv_integration_time,
                        configuration=configuration,
                        phasecentre=phasecentre,
                        polarisation_frame=polarisation_frame,
                        source=source,
                        meta=meta,
                        low_precision="float64",
                    )

                    # Need to reassign with correct dtype
                    vis_template = vis_template.assign(
                        {
                            "weight": vis_template.weight.astype(weight_dtype),
                            "flags": vis_template.flags.astype(flags_dtype),
                        }
                    )

                    vis_list.append(vis_template)

    return vis_list


def get_col_from_ms(
    msname: str,
    colname: str,
    start_time_idx: int,
    ntimes: int,
    num_baselines: int,
    ack=False,
    field_ids: List[int] = None,
    data_desc_ids: List[int] = None,
) -> List[npt.NDArray]:
    """
    Extract data from a specific column in a Measurement Set.

    This function reads a slice of data from the specified column, determined
    by a starting time index, a duration (number of times), and the number of
    baselines. It iterates over the specified Field IDs and Data Description
    IDs, returning the extracted data for each combination.

    Parameters
    ----------
    msname : str
        The file path to the Measurement Set.
    colname : str
        The name of the column to retrieve (e.g., "DATA", "UVW", "FLAG").
    start_time_idx : int
        The index of the starting time step to read. This is used to calculate
        the starting row offset: ``start_time_idx * num_baselines``.
    ntimes : int
        The number of time steps to read.
    num_baselines : int
        The number of baselines per time step. Used to calculate the total
        number of rows to read.
    ack : bool, optional
        If True, print an acknowledgement message when opening the table.
        Default is False.
    field_ids : list[int], optional
        A list of Field IDs to query. If None, defaults to [0].
    data_desc_ids : list[int], optional
        A list of Data Description IDs to query. If None, defaults to [0].

    Returns
    -------
    list[numpy.ndarray]
        A list of NumPy arrays containing the column data. Each element in the
        list corresponds to the data extracted for a specific combination of
        Field ID and Data Description ID.

    Raises
    ------
    ValueError
        If the query for a specific Field ID or Data Description ID returns
        zero rows (empty selection).
    """
    field_ids = field_ids or [0]
    data_desc_ids = data_desc_ids or [0]

    start_row = start_time_idx * num_baselines
    n_rows = ntimes * num_baselines

    col_data_per_field_dd = []

    with table(msname, ack=ack, readonly=True) as tab:
        for field in field_ids:
            with tab.query(f"FIELD_ID=={field}", style="") as ftab:
                if ftab.nrows() <= 0:
                    raise ValueError(f"Empty selection for FIELD_ID={field}")

                for dd in data_desc_ids:
                    with ftab.query(f"DATA_DESC_ID=={dd}", style="") as ms:
                        if ms.nrows() <= 0:
                            raise ValueError(
                                f"Empty selection for FIELD_ID= {field} "
                                f"and DATA_DESC_ID={dd}"
                            )
                        col_data = ms.getcol(
                            colname, startrow=start_row, nrow=n_rows
                        )
                        col_data_per_field_dd.append(col_data)

    return col_data_per_field_dd


def _load_vis_xdr(
    vis_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    vis_baseline_indices_to_update: np.ndarray,
    num_baselines_in_ms: int,
    crosscorr_mask_over_baseline: Optional[np.ndarray] = None,
    polarisation_order: Optional[np.ndarray] = None,
    datacolumn: str = "DATA",
    field_id: int = 0,
    data_desc_id: int = 0,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    vis_data_shape = (
        vis_chunk.shape[0],
        num_baselines_in_ms,
        *vis_chunk.shape[2:],
    )

    vis_data = get_col_from_ms(
        ms_name,
        colname=datacolumn,
        start_time_idx=start_time_idx,
        ntimes=ntimes,
        num_baselines=num_baselines_in_ms,
        field_ids=[field_id],
        data_desc_ids=[data_desc_id],
    )[0].reshape(vis_data_shape)

    if crosscorr_mask_over_baseline is not None:
        vis_data[:, crosscorr_mask_over_baseline, ...] = np.conj(
            vis_data[:, crosscorr_mask_over_baseline, ...]
        )

        if polarisation_order is not None:
            vis_data[:, crosscorr_mask_over_baseline, ...] = vis_data[
                :, crosscorr_mask_over_baseline, ...
            ][..., polarisation_order]

    actual_vis_data = np.zeros_like(vis_chunk)
    actual_vis_data[:, vis_baseline_indices_to_update, ...] = vis_data

    del vis_data

    return xr.DataArray(actual_vis_data, coords=vis_chunk.coords)


def _load_flags_xdr(
    flags_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    vis_baseline_indices_to_update: np.ndarray,
    num_baselines_in_ms: int,
    crosscorr_mask_over_baseline: Optional[np.ndarray] = None,
    polarisation_order: Optional[np.ndarray] = None,
    field_id: int = 0,
    data_desc_id: int = 0,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    flag_data_shape = (
        flags_chunk.shape[0],
        num_baselines_in_ms,
        *flags_chunk.shape[2:],
    )

    flag_data = get_col_from_ms(
        ms_name,
        colname="FLAG",
        start_time_idx=start_time_idx,
        ntimes=ntimes,
        num_baselines=num_baselines_in_ms,
        field_ids=[field_id],
        data_desc_ids=[data_desc_id],
    )[0].reshape(flag_data_shape)

    if (
        crosscorr_mask_over_baseline is not None
        and polarisation_order is not None
    ):
        flag_data[:, crosscorr_mask_over_baseline, ...] = flag_data[
            :, crosscorr_mask_over_baseline, ...
        ][..., polarisation_order]

    actual_flags_data = np.zeros_like(flags_chunk)
    actual_flags_data[:, vis_baseline_indices_to_update, ...] = flag_data

    del flag_data

    return xr.DataArray(actual_flags_data, coords=flags_chunk.coords)


def _load_weight_xdr(
    weight_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    vis_baseline_indices_to_update: np.ndarray,
    num_baselines_in_ms: int,
    crosscorr_mask_over_baseline: Optional[np.ndarray] = None,
    polarisation_order: Optional[np.ndarray] = None,
    field_id: int = 0,
    data_desc_id: int = 0,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    weight_data_shape = (
        weight_chunk.shape[0],
        num_baselines_in_ms,
        weight_chunk.shape[-1],
    )

    weight_data = get_col_from_ms(
        ms_name,
        colname="WEIGHT",
        start_time_idx=start_time_idx,
        ntimes=ntimes,
        num_baselines=num_baselines_in_ms,
        field_ids=[field_id],
        data_desc_ids=[data_desc_id],
    )[0].reshape(weight_data_shape)

    if (
        crosscorr_mask_over_baseline is not None
        and polarisation_order is not None
    ):
        weight_data[:, crosscorr_mask_over_baseline, ...] = weight_data[
            :, crosscorr_mask_over_baseline, ...
        ][..., polarisation_order]

    actual_weight_data = np.zeros_like(weight_chunk)
    actual_weight_data[:, vis_baseline_indices_to_update, ...] = weight_data[
        :, :, np.newaxis, ...
    ]

    del weight_data

    return xr.DataArray(actual_weight_data, coords=weight_chunk.coords)


def _load_uvw_xdr(
    uvw_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    vis_baseline_indices_to_update: np.ndarray,
    num_baselines_in_ms: int,
    crosscorr_mask_over_baseline: Optional[np.ndarray] = None,
    field_id: int = 0,
    data_desc_id: int = 0,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    uvw_data_shape = (uvw_chunk.shape[0], num_baselines_in_ms, 3)

    uvw_data = get_col_from_ms(
        ms_name,
        colname="UVW",
        start_time_idx=start_time_idx,
        ntimes=ntimes,
        num_baselines=num_baselines_in_ms,
        field_ids=[field_id],
        data_desc_ids=[data_desc_id],
    )[0].reshape(uvw_data_shape)

    # This sign switch was done in the original converter in data models
    uvw_data = -1 * uvw_data

    if crosscorr_mask_over_baseline is not None:
        uvw_data[:, crosscorr_mask_over_baseline, :] *= -1

    actual_uvw_data = np.zeros_like(uvw_chunk)
    actual_uvw_data[:, vis_baseline_indices_to_update, ...] = uvw_data

    del uvw_data

    return xr.DataArray(actual_uvw_data, coords=uvw_chunk.coords)


def _load_data_vars(
    vis: Visibility,
    ms_name: str,
    datacolumn: str = "DATA",
    field_id: int = 0,
    data_desc_id: int = 0,
):
    """
    Pre-requisites:

      * vis dimensions:
          time, baselineid, frequency, polarisation
        Measurement set "data" dimensions:
          rows (time * baselineid), frequency, polarisation

      * Measurement set may or may not contain auto-correlated values,
        but the visibility always expects auto-correlated values.
        Thus necessary conversions are made here. In case auto-corrs
        are absent in MS, auto-correlations in the Visibility dataset
        are set to zero.
    """
    with table(ms_name, readonly=True, ack=False) as tab:
        with tab.query(
            f"FIELD_ID=={field_id} AND DATA_DESC_ID=={data_desc_id}", style=""
        ) as ms:
            num_rows_in_ms = ms.nrows()
            if num_rows_in_ms <= 0:
                raise ValueError(
                    f"Empty selection for FIELD_ID={field_id} "
                    f"and DATA_DESC_ID={data_desc_id}"
                )

            ms_contains_autocorrelations = False
            if (
                taql(
                    "select ANTENNA1 from $1 where "
                    "ANTENNA1 == ANTENNA2 limit 1",
                    tables=[ms],
                ).nrows()
                > 0
            ):
                ms_contains_autocorrelations = True
            logger.info(
                "Does measurement set contain autocorrelations? %s",
                ms_contains_autocorrelations,
            )

            ms_is_baseline_order_reversed = False
            if (
                taql(
                    "select ANTENNA1 from $1 where "
                    "ANTENNA1 > ANTENNA2 limit 1",
                    tables=[ms],
                ).nrows()
                > 0
            ):
                ms_is_baseline_order_reversed = True
            logger.info(
                "In the measurement set, is the baseline antenna "
                "order reversed (i.e. is antenna1 > antenna2)? %s",
                ms_is_baseline_order_reversed,
            )

            if ms_is_baseline_order_reversed and (
                taql(
                    "select ANTENNA1 from $1 where "
                    "ANTENNA1 < ANTENNA2 limit 1",
                    tables=[ms],
                ).nrows()
                > 0
            ):
                raise RuntimeError(
                    "Order of antennas in baseline pairs is not consistent."
                )

    time_index_xdr = xr.DataArray(
        da.arange(vis.time.size), coords={"time": vis.time}
    ).pipe(with_chunks, vis.chunksizes)

    nantennas = vis.configuration.id.size

    # Visibility always has baselines with autocorrelations,
    # and order antenna1 <= antenna2
    vis_baseline_indices = pandas.MultiIndex.from_arrays(
        np.triu_indices(nantennas, k=0), names=("antenna1", "antenna2")
    )

    if ms_is_baseline_order_reversed:
        ms_baseline_indices_order = slice(None, None, -1)
    else:
        ms_baseline_indices_order = slice(None, None, None)

    if ms_contains_autocorrelations:
        diag_offset = 0
    else:
        diag_offset = 1

    ms_baseline_indices = pandas.MultiIndex.from_arrays(
        np.triu_indices(nantennas, k=diag_offset)[ms_baseline_indices_order],
        names=("antenna1", "antenna2"),
    )

    num_baselines_in_ms = num_rows_in_ms // time_index_xdr.size

    assert num_baselines_in_ms == len(ms_baseline_indices), (
        "Number of baselines in measurement set (%s) do not match with "
        "number of baselines from indices (%s)",
        num_baselines_in_ms,
        len(ms_baseline_indices),
    )

    vis_baseline_indices_to_update = np.array(
        [
            vis_baseline_indices.get_loc(indices[ms_baseline_indices_order])
            for indices in ms_baseline_indices
        ]
    )

    crosscorr_baseline_mask = None
    polarisation_order = None
    if ms_is_baseline_order_reversed:
        crosscorr_baseline_mask = ms_baseline_indices.get_level_values(
            "antenna1"
        ) != ms_baseline_indices.get_level_values("antenna2")

        if vis._polarisation_frame in ["linear", "circular"]:
            polarisation_order = [0, 2, 1, 3]
        elif vis._polarisation_frame == "linearFITS":
            polarisation_order = [0, 1, 3, 2]
        else:
            raise RuntimeError(
                "Unsupported polarisation frame '%s' "
                "when antenna order in baselines is reversed",
                vis._polarisation_frame,
            )

    # vis
    vis_data_xdr = xr.map_blocks(
        _load_vis_xdr,
        vis.vis,
        args=[
            ms_name,
            time_index_xdr,
            vis_baseline_indices_to_update,
            num_baselines_in_ms,
            crosscorr_baseline_mask,
            polarisation_order,
            datacolumn,
            field_id,
            data_desc_id,
        ],
        template=vis.vis,
    )

    # flags
    flag_data_xdr = xr.map_blocks(
        _load_flags_xdr,
        vis.flags,
        args=[
            ms_name,
            time_index_xdr,
            vis_baseline_indices_to_update,
            num_baselines_in_ms,
            crosscorr_baseline_mask,
            polarisation_order,
            field_id,
            data_desc_id,
        ],
        template=vis.flags,
    )

    # weight
    weight_data_xdr = xr.map_blocks(
        _load_weight_xdr,
        vis.weight,
        args=[
            ms_name,
            time_index_xdr,
            vis_baseline_indices_to_update,
            num_baselines_in_ms,
            crosscorr_baseline_mask,
            polarisation_order,
            field_id,
            data_desc_id,
        ],
        template=vis.weight,
    )

    # uvw
    uvw_data_xdr = xr.map_blocks(
        _load_uvw_xdr,
        vis.uvw,
        args=[
            ms_name,
            time_index_xdr,
            vis_baseline_indices_to_update,
            num_baselines_in_ms,
            crosscorr_baseline_mask,
            field_id,
            data_desc_id,
        ],
        template=vis.uvw,
    )

    return vis.assign(
        {
            "vis": vis_data_xdr,
            "flags": flag_data_xdr,
            "weight": weight_data_xdr,
            "uvw": uvw_data_xdr,
        }
    )


def load_ms_as_dataset_with_time_chunks(
    ms_name: str,
    times_per_chunk: int,
    ack: bool = False,
    datacolumn: str = "DATA",
    field_id: int = 0,
    data_desc_id: int = 0,
) -> Visibility:
    """
    Load MSv2 data into a Visibility dataset using distributed time chunks.

    This function loads data for a specific field and data description ID into
    a Visibility object. The loading is distributed, chunking the data along
    the time axis to facilitate parallel processing (e.g., with Dask).

    Parameters
    ----------
    ms_name : str
        The file path to the Measurement Set.
    times_per_chunk : int
        The number of time steps to include in each Dask chunk.
    ack : bool, optional
        If True, print an acknowledgement message when opening the table.
        Default is False.
    datacolumn : str, optional
        The name of the column to read (e.g., "DATA"). Default is "DATA".
    field_id : int, optional
        The Field ID to load. Default is 0.
    data_desc_id : int, optional
        The Data Description ID to load. Default is 0.

    Returns
    -------
    Visibility
        The loaded Visibility dataset with dask-backed arrays.

    Notes
    -----
    The `baselines` dimension in the returned dataset is simplified to a
    NumPy array of baseline IDs, rather than the standard Pandas MultiIndex
    used by the Visibility class. This modification is necessary because
    `xarray` operations like `map_blocks` do not support Pandas MultiIndex
    coordinates.

    **Important:** You must restore the baselines to the original Pandas
    MultiIndex format before passing this object to any functions in
    `ska-sdp-func-python`.
    """
    # Get observation metadata
    vis_template = simplify_baselines_dim(
        create_template_vis_from_ms(
            ms_name,
            ack=ack,
            datacolumn=datacolumn,
            field_ids=[field_id],
            data_desc_ids=[data_desc_id],
        )[0]
    )

    chunks = {
        "time": times_per_chunk,
        "baselineid": -1,
        "frequency": -1,
        "polarisation": -1,
        "spatial": -1,
    }

    vis_template = vis_template.pipe(with_chunks, chunks)

    return _load_data_vars(
        vis_template, ms_name, datacolumn, field_id, data_desc_id
    )


def _generate_file_paths_for_vis_zarr_file(vis_cache_directory):
    attributes_file = os.path.join(vis_cache_directory, ATTRS_FILE_NAME)
    baselines_file = os.path.join(vis_cache_directory, BASELINE_FILE_NAME)
    vis_zarr_file = os.path.join(vis_cache_directory, VIS_FILE_NAME)

    return attributes_file, baselines_file, vis_zarr_file


def write_ms_to_zarr(
    input_ms_path,
    vis_cache_directory,
    zarr_chunks,
    ack=False,
    datacolumn="DATA",
    field_id: int = 0,
    data_desc_id: int = 0,
):
    """
    Convert a MSv2 into a Visibility dataset and write it to zarr.
    NOTE: The baselines coordinates in Visibility are simplified.
    See note section in :py:func:`load_ms_as_dataset_with_time_chunks`
    """
    visibility = load_ms_as_dataset_with_time_chunks(
        input_ms_path,
        zarr_chunks["time"],
        ack=ack,
        datacolumn=datacolumn,
        field_id=field_id,
        data_desc_id=data_desc_id,
    )

    writer = write_visibility_to_zarr(
        vis_cache_directory, zarr_chunks, visibility
    )

    logger.warning("Triggering eager compute to dump visibilities to zarr.")
    dask.compute(writer)


def write_visibility_to_zarr(
    directory_to_write, zarr_chunks, visibility: Visibility
) -> Delayed:
    """
    Writes Visibility to zarr file in the provided directory.

    Since native xarray.to_zarr() function does not allow writing
    python-object like attributes and coordinates, this function
    first writes the attributes and "baselines" coordinate values as
    python pickeled files, and removed them from visibility.
    Then writes the rest of the visibility to a zarr file.

    Returns
    -------
    dask.delayed
        Returns a dask delayed zarr writer task which the user
        needs to call compute on to write the actual visibilities.
    """
    attributes_file, baselines_file, vis_zarr_file = (
        _generate_file_paths_for_vis_zarr_file(directory_to_write)
    )

    attrs = deepcopy(visibility.attrs)
    with open(attributes_file, "wb") as file:
        pickle.dump(attrs, file)

    baselines = deepcopy(visibility.baselines).compute()
    with open(baselines_file, "wb") as file:
        pickle.dump(baselines, file)

    writer = (
        visibility.drop_attrs()
        .drop_vars("baselines")
        .pipe(with_chunks, zarr_chunks)
        .to_zarr(vis_zarr_file, mode="w", compute=False)
    )

    return writer


def read_visibility_from_zarr(vis_cache_directory, vis_chunks) -> Visibility:
    """
    Read a Visibility dataset from a Zarr cache directory.

    This function reconstructs a Visibility object by opening the main Zarr
    storage and manually reloading metadata that cannot be natively stored in
    Zarr (such as complex object attributes and Pandas MultiIndex baselines)
    from separate pickle files.

    Parameters
    ----------
    vis_cache_directory : str
        The path to the directory containing the cached Zarr store and
        associated metadata pickle files.
    vis_chunks : dict
        The chunking scheme to apply when opening the dataset (e.g.,
        ``{'time': 1, 'frequency': 10}``). Passed directly to
        ``xr.open_dataset``.

    Returns
    -------
    Visibility
        The fully reconstructed Visibility dataset with attributes and
        baseline coordinates restored.
    """
    attributes_file, baselines_file, vis_zarr_file = (
        _generate_file_paths_for_vis_zarr_file(vis_cache_directory)
    )

    zarr_data = xr.open_dataset(
        vis_zarr_file, chunks=vis_chunks, engine="zarr"
    )

    with open(attributes_file, "rb") as file:
        attrs = pickle.load(file)

    zarr_data = zarr_data.assign_attrs(attrs)

    with open(baselines_file, "rb") as file:
        baselines = pickle.load(file)

    zarr_data = zarr_data.assign({"baselines": baselines})

    # Explictly load antenna1 and antenna2 coordinates
    zarr_data.antenna1.load()
    zarr_data.antenna2.load()

    return zarr_data


def check_if_cache_files_exist(vis_cache_directory):
    """
    Verify if the necessary cache files exist in the specified directory.

    This function checks for the presence of three specific artifacts required
    to reconstruct a Visibility dataset: the attributes pickle file, the
    baselines pickle file, and the Zarr directory itself.

    Parameters
    ----------
    vis_cache_directory : str
        The path to the directory to inspect.

    Returns
    -------
    bool
        True if all required files and directories exist; False otherwise.
    """
    attributes_file, baselines_file, vis_zarr_file = (
        _generate_file_paths_for_vis_zarr_file(vis_cache_directory)
    )

    return (
        os.path.isfile(attributes_file)
        and os.path.isfile(baselines_file)
        and os.path.isdir(vis_zarr_file)
    )
