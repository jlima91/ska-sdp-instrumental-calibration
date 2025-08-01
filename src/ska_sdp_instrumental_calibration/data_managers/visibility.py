import logging
import os
import pickle
from copy import deepcopy
from typing import List

import dask
import dask.array as da
import numpy
import numpy as np
import pandas
import xarray as xr
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from casacore.tables import table
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    simplify_baselines_dim,
)

from ..workflow.utils import with_chunks

logger = logging.getLogger(__name__)

ATTRS_FILE_NAME = "attrs.pickle"
VIS_FILE_NAME = "vis.zarr"
BASELINE_FILE_NAME = "baselines.pickle"


def create_template_vis_from_ms(
    msname,
    ack=False,
    datacolumn="DATA",
) -> List[Visibility]:
    """
    Creates empty "template" visibility objects from given
    MS file path.

    Returns a list of Visibility, where each visibility's
    vis, flags and weights are empty dask arrays with
    correct shapes and dtypes. These can be filled in later
    by reading the actual data.

    Number of elements in the returned list correspond to
    field ids and data desc ids.
    """
    # pylint: disable=import-outside-toplevel
    vis_list = []
    with table(msname, ack=ack, readonly=True) as tab:
        fields = numpy.unique(tab.getcol("FIELD_ID"))
        dds = numpy.unique(tab.getcol("DATA_DESC_ID"))

        for field in fields:
            with tab.query(f"FIELD_ID=={field}", style="") as ftab:
                if ftab.nrows() <= 0:
                    raise ValueError(f"Empty selection for FIELD_ID={field}")

                for dd in dds:
                    # Now get info from the subtables
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
                    start_time = numpy.min(time) / 86400.0
                    end_time = numpy.max(time) / 86400.0

                    logger.debug(
                        "create_visibility_from_ms: Observation from %s to %s",
                        Time(start_time, format="mjd").iso,
                        Time(end_time, format="mjd").iso,
                    )

                    with table(
                        f"{msname}/SPECTRAL_WINDOW", ack=ack, readonly=True
                    ) as spwtab:
                        cfrequency = numpy.array(
                            spwtab.getcol("CHAN_FREQ")[spwid]
                        )
                        cchannel_bandwidth = numpy.array(
                            spwtab.getcol("CHAN_WIDTH")[spwid]
                        )

                    nchan = cfrequency.shape[0]

                    # Get polarisation info
                    with table(
                        f"{msname}/POLARIZATION", ack=ack, readonly=True
                    ) as poltab:
                        corr_type = poltab.getcol("CORR_TYPE")[polid]

                    # These correspond to the CASA Stokes enumerations
                    if numpy.array_equal(corr_type, [1, 2, 3, 4]):
                        polarisation_frame = PolarisationFrame("stokesIQUV")
                        npol = 4
                    elif numpy.array_equal(corr_type, [1, 2]):
                        polarisation_frame = PolarisationFrame("stokesIQ")
                        npol = 2
                    elif numpy.array_equal(corr_type, [1, 4]):
                        polarisation_frame = PolarisationFrame("stokesIV")
                        npol = 2
                    elif numpy.array_equal(corr_type, [5, 6, 7, 8]):
                        polarisation_frame = PolarisationFrame("circular")
                        npol = 4
                    elif numpy.array_equal(corr_type, [5, 8]):
                        polarisation_frame = PolarisationFrame("circularnp")
                        npol = 2
                    elif numpy.array_equal(corr_type, [9, 10, 11, 12]):
                        polarisation_frame = PolarisationFrame("linear")
                        npol = 4
                    elif numpy.array_equal(corr_type, [9, 12, 10, 11]):
                        polarisation_frame = PolarisationFrame("linearFITS")
                        npol = 4
                    elif numpy.array_equal(corr_type, [9, 12]):
                        polarisation_frame = PolarisationFrame("linearnp")
                        npol = 2
                    elif numpy.array_equal(
                        corr_type, [9]
                    ) or numpy.array_equal(corr_type, [1]):
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
                        names = numpy.array(anttab.getcol("NAME"))

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
                            names = numpy.repeat("No name", len(names))

                        mount = numpy.array(anttab.getcol("MOUNT"))[
                            names != ""
                        ]
                        # logger.info("mount is: %s" % (mount))
                        diameter = numpy.array(anttab.getcol("DISH_DIAMETER"))[
                            names != ""
                        ]
                        xyz = numpy.array(anttab.getcol("POSITION"))[
                            names != ""
                        ]
                        offset = numpy.array(anttab.getcol("OFFSET"))[
                            names != ""
                        ]
                        stations = numpy.array(anttab.getcol("STATION"))[
                            names != ""
                        ]
                        names = numpy.array(anttab.getcol("NAME"))[names != ""]
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

                    time_index_row = numpy.zeros_like(time, dtype="int")
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
                        numpy.unique(time_index_row)
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

                    bv_times = numpy.zeros([ntimes])
                    bv_integration_time = numpy.zeros([ntimes])

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
) -> List[np.array]:
    """
    Get data from a column in measurment set.
    The data is read from partial rows based on start_time_idx,
    number of times and number of baselines.

    Returns a list of np arrays, where each element
    corresponds to corresponding column data
    from one field and one data description
    """
    start_row = start_time_idx * num_baselines
    n_rows = ntimes * num_baselines

    col_data_per_field_dd = []

    with table(msname, ack=ack, readonly=True) as tab:
        fields = numpy.unique(tab.getcol("FIELD_ID"))
        dds = numpy.unique(tab.getcol("DATA_DESC_ID"))

        for field in fields:
            with tab.query(f"FIELD_ID=={field}", style="") as ftab:
                if ftab.nrows() <= 0:
                    raise ValueError(f"Empty selection for FIELD_ID={field}")

                for dd in dds:
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
    baseline_indices_pair: np.ndarray,
    num_baselines_in_ms: int,
    datacolumn: str = "DATA",
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    vis_non_corr_shape = (
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
    )[0].reshape(vis_non_corr_shape)

    actual_vis_data = np.zeros_like(vis_chunk)
    actual_vis_data[:, baseline_indices_pair[:, 0], ...] = vis_data[
        :, baseline_indices_pair[:, 1], ...
    ]

    del vis_data

    return xr.DataArray(actual_vis_data, coords=vis_chunk.coords)


def _load_flags_xdr(
    flags_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    baseline_indices_pair: np.ndarray,
    num_baselines_in_ms: int,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    flag_non_corr_shape = (
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
    )[0].reshape(flag_non_corr_shape)

    actual_flags_data = np.zeros_like(flags_chunk)
    actual_flags_data[:, baseline_indices_pair[:, 0], ...] = flag_data[
        :, baseline_indices_pair[:, 1], ...
    ]

    del flag_data

    return xr.DataArray(actual_flags_data, coords=flags_chunk.coords)


def _load_weight_xdr(
    weight_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    baseline_indices_pair: np.ndarray,
    num_baselines_in_ms: int,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    weight_non_corr_shape = (
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
    )[0].reshape(weight_non_corr_shape)

    actual_weight_data = np.zeros_like(weight_chunk)
    actual_weight_data[:, baseline_indices_pair[:, 0], ...] = weight_data[
        :, baseline_indices_pair[:, 1], np.newaxis, ...
    ]

    del weight_data

    return xr.DataArray(actual_weight_data, coords=weight_chunk.coords)


def _load_uvw_xdr(
    uvw_chunk: xr.DataArray,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    baseline_indices_pair: np.ndarray,
    num_baselines_in_ms: int,
):
    start_time_idx = time_index_xdr.data[0]
    ntimes = time_index_xdr.size

    uvw_non_corr_shape = (uvw_chunk.shape[0], num_baselines_in_ms, 3)

    uvw_data = get_col_from_ms(
        ms_name,
        colname="UVW",
        start_time_idx=start_time_idx,
        ntimes=ntimes,
        num_baselines=num_baselines_in_ms,
    )[0].reshape(uvw_non_corr_shape)

    # This sign switch was done in the original converter in data models
    uvw_data = -1 * uvw_data

    actual_uvw_data = np.zeros_like(uvw_chunk)
    actual_uvw_data[:, baseline_indices_pair[:, 0], ...] = uvw_data[
        :, baseline_indices_pair[:, 1], ...
    ]

    del uvw_data

    return xr.DataArray(actual_uvw_data, coords=uvw_chunk.coords)


def _load_data_vars(
    vis: Visibility,
    ms_name: str,
    time_index_xdr: xr.DataArray,
    datacolumn: str = "DATA",
):
    """
    Pre-requisites:
      * vis dimensions:
          time, baselineid, frequency, polarisation
          Measurement set "data" dimensions:
          rows (time , baselineid), frequency, polarisation

      * Measurement set may or may not contain self-correlated values,
        but the visibility always expects self-correlated values.
        Thus necessary conversions are made here. In case self-corrs
        are absent in MS, self-correlations in the Visibility dataset
        are set to zero.

      * In a baseline, Antenna1 index is always less than or equal
        to Antenna2 index.

    Assumptions:
      * Measurement set contains only one field id and one data
        description id.
    """
    nantennas = vis.configuration.id.size
    ntime = time_index_xdr.size

    baselines_self_corr = pandas.MultiIndex.from_arrays(
        np.triu_indices(nantennas, k=0), names=("antenna1", "antenna2")
    )
    baselines_no_self_corr = pandas.MultiIndex.from_arrays(
        np.triu_indices(nantennas, k=1), names=("antenna1", "antenna2")
    )

    with table(ms_name, readonly=True, ack=False) as ms:
        num_rows = ms.nrows()

    # MS contains self-correlated visibilities
    if (ntime * nantennas * (nantennas + 1) // 2) == num_rows:
        baseline_indices_pair = np.array(
            [[i, i] for i in range(len(baselines_self_corr))]
        )
        num_baselines_in_ms = len(baselines_self_corr)
    else:  # MS does not contain self-correlated visibilities
        baseline_indices_pair = np.array(
            [
                [baselines_self_corr.get_loc((ant1, ant2)), no_self_corr_row]
                for no_self_corr_row, (ant1, ant2) in enumerate(
                    baselines_no_self_corr
                )
            ]
        )
        num_baselines_in_ms = len(baselines_no_self_corr)

    # vis
    vis_data_xdr = xr.map_blocks(
        _load_vis_xdr,
        vis.vis,
        args=[
            ms_name,
            time_index_xdr,
            baseline_indices_pair,
            num_baselines_in_ms,
            datacolumn,
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
            baseline_indices_pair,
            num_baselines_in_ms,
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
            baseline_indices_pair,
            num_baselines_in_ms,
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
            baseline_indices_pair,
            num_baselines_in_ms,
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
) -> xr.Dataset:
    """
    Distributed load of a MSv2 Measurement Set into a Visibility dataset,
    across time chunks.

    Assumptions:
        Measurement set contains only one field id and one data description id.
    """
    # Get observation metadata
    vis_template = simplify_baselines_dim(
        create_template_vis_from_ms(
            ms_name,
            ack=ack,
            datacolumn=datacolumn,
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

    time_index_xdr = xr.DataArray(
        da.arange(vis_template.time.size), coords={"time": vis_template.time}
    ).pipe(with_chunks, chunks)

    return _load_data_vars(vis_template, ms_name, time_index_xdr, datacolumn)


def write_ms_to_zarr(input_ms_path, vis_cache_directory, zarr_chunks):
    attributes_file = os.path.join(vis_cache_directory, ATTRS_FILE_NAME)
    baselines_file = os.path.join(vis_cache_directory, BASELINE_FILE_NAME)
    vis_zarr_file = os.path.join(vis_cache_directory, VIS_FILE_NAME)

    data: xr.Dataset = load_ms_as_dataset_with_time_chunks(
        input_ms_path, zarr_chunks["time"]
    )

    attrs = deepcopy(data.attrs)
    with open(attributes_file, "wb") as file:
        pickle.dump(attrs, file)

    baselines = deepcopy(data.baselines).compute()
    with open(baselines_file, "wb") as file:
        pickle.dump(baselines, file)

    writer = (
        data.drop_attrs()
        .drop_vars("baselines")
        .pipe(with_chunks, zarr_chunks)
        .to_zarr(vis_zarr_file, mode="w", compute=False)
    )

    logger.warning("Triggering eager compute to dump visibilities to zarr.")
    dask.compute(writer)


def check_if_cache_files_exist(vis_cache_directory):
    return (
        os.path.isfile(os.path.join(vis_cache_directory, ATTRS_FILE_NAME))
        and os.path.isfile(
            os.path.join(vis_cache_directory, BASELINE_FILE_NAME)
        )
        and os.path.isdir(os.path.join(vis_cache_directory, VIS_FILE_NAME))
    )


def read_dataset_from_zarr(vis_cache_directory, vis_chunks):

    attributes_file = os.path.join(vis_cache_directory, ATTRS_FILE_NAME)
    baselines_file = os.path.join(vis_cache_directory, BASELINE_FILE_NAME)
    vis_zarr_file = os.path.join(vis_cache_directory, VIS_FILE_NAME)

    zarr_data = xr.open_dataset(
        vis_zarr_file, chunks=vis_chunks, engine="zarr"
    )

    with open(attributes_file, "rb") as file:
        attrs = pickle.load(file)

    zarr_data = zarr_data.assign_attrs(attrs)

    with open(baselines_file, "rb") as file:
        baselines = pickle.load(file)

    zarr_data = zarr_data.assign({"baselines": baselines})

    return zarr_data
