"""Helper functions"""

__all__ = [
    "create_demo_ms",
    "create_bandpass_table",
    "get_ms_metadata",
    "get_phasecentre",
    "create_soltab_group",
    "create_soltab_datasets",
    "convert_model_to_skycomponents",
]

import warnings
from collections import namedtuple
from typing import Literal

import h5py
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from casacore.tables import table
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms

from ska_sdp_instrumental_calibration.logger import setup_logger

from .utils_dep import (
    convert_model_to_skycomponents,
    create_bandpass_table,
    create_demo_ms,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = setup_logger(__name__)


def get_phasecentre(ms_name: str) -> SkyCoord:
    """Return the phase centre of a MSv2 Measurement Set.

    The first field is used if there more than one.

    :param ms_name: Name of input Measurement Set.
    :return: phase centre
    """
    fieldtab = table(f"{ms_name}/FIELD", ack=False)
    field = 0
    pc = fieldtab.getcol("PHASE_DIR")[field, 0, :]
    return SkyCoord(
        ra=pc[0], dec=pc[1], unit="radian", frame="icrs", equinox="J2000"
    )


def get_ms_metadata(
    ms_name: str,
    ack: bool = False,
    start_chan: int = 0,
    end_chan: int = 0,
    datacolumn: str = "DATA",
    selected_sources: list = None,
    selected_dds: list = None,
    average_channels: bool = False,
) -> xr.Dataset:
    """Get Visibility dataset metadata.

    Fixme: use ska_sdp_datamodels.visibility.vis_io_ms.get_ms_metadata once
    YAN-1990 is finalised. For now, read a single channel and use its metadata.

    :param ms_name: Name of input Measurement Set
    :param ack: Ask casacore to acknowledge each table operation
    :param start_chan: Starting channel to read
    :param end_chan: End channel to read4
    :param datacolumn: MS data column to read DATA, CORRECTED_DATA, MODEL_DATA
    :param selected_sources: Sources to select
    :param selected_dds: Data descriptors to select
    :param average_channels: Average all channels read
    :return: Namedtuple of metadata products required by Visibility.constructor
        - uvw
        - baselines
        - time
        - frequency
        - channel_bandwidth
        - integration_time
        - configuration
        - phasecentre
        - polarisation_frame
        - source
        - meta
    """
    # Read a single-channel from the dataset
    tmpvis = create_visibility_from_ms(
        ms_name,
        start_chan=start_chan,
        ack=ack,
        datacolumn=datacolumn,
        end_chan=end_chan,
        selected_sources=selected_sources,
        selected_dds=selected_dds,
        average_channels=average_channels,
    )[0]
    # Update frequency metadata for the full dataset
    spwtab = table(f"{ms_name}/SPECTRAL_WINDOW", ack=False)
    frequency = np.array(spwtab.getcol("CHAN_FREQ")[0])
    channel_bandwidth = np.array(spwtab.getcol("CHAN_WIDTH")[0])

    ms_metadata = namedtuple(
        "ms_metadata",
        [
            "uvw",
            "baselines",
            "time",
            "frequency",
            "channel_bandwidth",
            "integration_time",
            "configuration",
            "phasecentre",
            "polarisation_frame",
            "source",
            "meta",
        ],
    )

    return ms_metadata(
        uvw=tmpvis.uvw.data,
        baselines=tmpvis.baselines,
        time=tmpvis.time,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        integration_time=tmpvis.integration_time,
        configuration=tmpvis.configuration,
        phasecentre=tmpvis.phasecentre,
        polarisation_frame=PolarisationFrame(tmpvis._polarisation_frame),
        source="bpcal",
        meta=None,
    )


def create_soltab_group(
    solset: h5py.Group, solution_type: Literal["amplitude", "phase", "clock"]
) -> h5py.Group:
    """Create soltab group under given solset group.

    :param solset: base-level HDF5 group to update
    :param solution_type: only "amplitude" and "phase" are supported at present
    :return: HDF5 group for the "solution_type" data
    """
    soltab = solset.create_group(f"{solution_type}000")
    soltab.attrs["TITLE"] = np.bytes_(solution_type)
    return soltab


def create_soltab_datasets(soltab: h5py.Group, gaintable: GainTable):
    """Add a dataset for each of the GainTable dimensions.

    :param soltab: HDF5 table to update
    :param gaintable: GainTable
    """
    # create a dataset for each dimension
    for dim in list(gaintable.gain.sizes):
        soltab.create_dataset(dim, data=gaintable[dim].data)

    # create datasets for the data and weights
    shape = gaintable.gain.shape
    axes = np.bytes_(",".join(list(gaintable.gain.sizes)))

    val = soltab.create_dataset("val", shape=shape, dtype=float)
    val.attrs["AXES"] = axes

    weight = soltab.create_dataset("weight", shape=shape, dtype=float)
    weight.attrs["AXES"] = axes

    return val, weight


def create_clock_soltab_datasets(soltab: h5py.Group, delaytable: xr.Dataset):
    """Add a dataset for each of the Delay dimensions.

    :param soltab: HDF5 table to update
    :param delaytable: xr.Dataset
    """
    # create a dataset for each dimension
    for dim in list(delaytable.delay.sizes):
        soltab.create_dataset(dim, data=delaytable[dim].data)

    # create datasets for the data and weights
    shape = delaytable.delay.shape
    axes = np.bytes_(",".join(list(delaytable.delay.sizes)))

    val = soltab.create_dataset("val", shape=shape, dtype=float)
    val.attrs["AXES"] = axes

    offset = soltab.create_dataset("offset", shape=shape, dtype=float)
    offset.attrs["AXES"] = axes

    return val, offset


def with_chunks(dataarray: xr.DataArray, chunks: dict) -> xr.DataArray:
    """
    Rechunk a DataArray along dimensions specified in `chunks` dict.

    Parameters
    ----------
    dataarray : xarray.DataArray
        Input DataArray (can be Dask-backed or not).
    chunks: dict
        A dictionary mapping dimension names to chunk sizes.

    Returns
    -------
    xarray.DataArray
        Rechunked DataArray if applicable.
    """
    relevant_chunks = {
        dim: chunks[dim] for dim in dataarray.dims if dim in chunks
    }

    return dataarray.chunk(relevant_chunks) if relevant_chunks else dataarray


def normalize_data(data):
    """
    Scales array data to the [0, 1] range, ignoring NaN values.

    This function performs min-max normalization on a *copy* of the
    input array. The minimum non-NaN value is mapped to 0 and the
    maximum non-NaN value is mapped to 1. NaN values are left unchanged.

    Parameters
    ----------
    data : numpy.ndarray
        The input array containing numerical data to be normalized.
        This array is *not* modified in-place.

    Returns
    -------
    numpy.ndarray
        A new array with non-NaN values scaled to the [0, 1] range.
        If the input array is empty or contains only NaN values,
        a copy of the original array is returned.
    """

    return data / np.linalg.norm(data, ord=1)
