"""
Delay tables written by the delay stage, one per measurement set: their
paths, looked up in the run metadata of a field, and their reader, which
reads them back for the QA report as one delay table per field. Written by
``export_clock_to_h5parm`` as a ``sol000/clock000`` soltab of per station,
per polarisation delays in seconds and phase offsets in cycles, of axes
``time,ant,pol``.
"""

import h5py
import numpy as np
import xarray as xr

from ...xarray_processors.delay import DelayTable
from ..exceptions import RunMetadataError
from .field_metadata import FieldMetadata
from .manifest_schema import ManifestSchema

_CLOCK_SOLTAB = "sol000/clock000"


# Axes of export_clock_to_h5parm, written without squeezing
_CLOCK_AXES = "time,ant,pol"


def get_delaytable_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the delay tables of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one delay table recorded per MS
        prefix.

    Returns
    -------
    list[str]
        Delay table of each measurement set of the field. Empty if none is
        recorded: the reader raises for it.
    """
    delaytables = (
        field_metadata.get_data_for(ManifestSchema.DELAY.DELAYTABLE) or {}
    )
    return list(delaytables.values())


def read_delaytable(delaytable_paths: list[str]) -> xr.Dataset:
    """
    Read the delay tables of the measurement sets of a field, written by
    ``export_clock_to_h5parm``, as one delay table.

    The measurement sets of a field split its observation in time, so
    their delay tables are concatenated along time, as the gaintables are
    on export, and ordered by time.

    Parameters
    ----------
    delaytable_paths : list[str]
        H5parm delay table of each measurement set of the field.

    Returns
    -------
    DelayTable
        Delays in seconds and offsets in cycles, of dimensions (time,
        antenna_name, pol).

    Raises
    ------
    RunMetadataError
        If there are no delay tables, one was not produced, cannot be read
        or is not of axes ``time,ant,pol``, or they do not have the same
        stations.
    """
    if not delaytable_paths:
        raise RunMetadataError("No delay table recorded")

    delaytables = [_read_ms_delaytable(path) for path in delaytable_paths]
    try:
        # join="exact": the measurement sets of a field share stations,
        # so tables of different stations are a bad product, not padded
        delaytable = xr.concat(delaytables, dim="time", join="exact")
    except ValueError as error:
        raise RunMetadataError(
            f"Delay tables {delaytable_paths} do not have the same "
            f"stations: {error}"
        ) from error

    # The measurement sets are not recorded in time order
    return delaytable.sortby("time")


def _read_ms_delaytable(delaytable_path: str) -> DelayTable:
    """
    Read the delay table of one measurement set.

    Parameters
    ----------
    delaytable_path : str
        Path of the H5parm delay table, written by
        ``export_clock_to_h5parm``.

    Returns
    -------
    DelayTable
        Delays in seconds and offsets in cycles, of dimensions (time,
        antenna_name, pol), with the station names stripped of the null
        character the export appends.

    Raises
    ------
    RunMetadataError
        If the delay table was not produced, cannot be read (it is
        corrupt, or has no clock soltab or one of its datasets), or is not
        of axes ``time,ant,pol``.
    """
    try:
        with h5py.File(delaytable_path, "r") as h5f:
            soltab = h5f[_CLOCK_SOLTAB]
            axes = _decode(soltab["val"].attrs["AXES"])
            if axes != _CLOCK_AXES:
                raise RunMetadataError(
                    f"Delay table {delaytable_path} has axes {axes}, "
                    f"not {_CLOCK_AXES}"
                )

            return DelayTable.constructor(
                delay=soltab["val"][...],
                offset=soltab["offset"][...],
                time=soltab["time"][...],
                antenna_name=np.array(
                    [_decode(name) for name in soltab["ant"][...]]
                ),
                pol=[_decode(pol) for pol in soltab["pol"][...]],
            )
    except FileNotFoundError as error:
        # The run metadata lists the products a run should write, so a
        # listed delay table can be missing if its export failed
        raise RunMetadataError(
            f"Delay table {delaytable_path} was not produced"
        ) from error
    except (OSError, KeyError) as error:
        # A bad product, not a bug: a truncated or corrupt HDF5 file
        # (OSError), or no clock soltab or one of its datasets (KeyError)
        raise RunMetadataError(
            f"Delay table {delaytable_path} cannot be read: {error}"
        ) from error


def _decode(value: bytes | str) -> str:
    """
    Decode a string of an H5parm attribute or coordinate.

    Parameters
    ----------
    value : bytes or str
        Value as stored by h5py: bytes for the null terminated names and
        labels written by the export.

    Returns
    -------
    str
        Value without the trailing null character the export adds to
        names.
    """
    text = value.decode() if isinstance(value, bytes) else str(value)
    return text.rstrip("\x00")
