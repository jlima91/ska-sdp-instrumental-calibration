import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("xarray_processors._utils")


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


def simplify_baselines_dim(vis: xr.Dataset) -> xr.Dataset:
    """Move the baselines coord to a data variable and use an index instead.

    A number of xarray/dask operations exit in error, complaining about the
    pandas.MultiIndex baselines coordinate of Visibility datasets. In most
    cases this can be avoided by replacing the baselines coordinate with a
    more simple coordinate and resetting baselines as a data variable.

    :param vis: Standard Visibility dataset
    :return: Modified Visibility dataset
    """
    if vis.coords.get("baselines") is None:
        logger.warning("No baselines coord in dataset. Returning unchanged")
        return vis
    else:
        logger.debug("Swapping baselines MultiIndex coord with indices")
        if vis.variables.get("baselineid") is None:
            vis = vis.assign_coords(
                baselineid=("baselines", np.arange(len(vis.baselines)))
            )
        return vis.swap_dims({"baselines": "baselineid"}).reset_coords(
            ("baselines", "antenna1", "antenna2")
        )


def parse_antenna(refant, station_names: xr.DataArray, station_counts: int):
    """
    Checks and converts a reference antenna identifier (index or name) to its
    corresponding index.

    refant : int or str
        Reference antenna, specified either as an integer index or as a string
        name.
    station_names : xr.DataArray
        Array of station names, used to map string names to indices.
    station_counts : int
        Total number of stations.

    int
        The index of the reference antenna.

    Raises
    ------
    ValueError
        If the reference antenna name or index is not valid.
    """

    if type(refant) is str:
        try:
            station_index = station_names.where(
                station_names == refant, drop=True
            ).id.values[0]
        except IndexError:
            raise ValueError("Reference antenna name is not valid")
        return station_index
    elif type(refant) is int:
        if refant > station_counts - 1 or refant < 0:
            raise ValueError("Reference antenna index is not valid")
        else:
            return refant

    raise RuntimeError(
        "Unsupported type for antenna. Only int or string supported"
    )
