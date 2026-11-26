"""
Gaintables written by the pipeline as H5parm files: their paths, looked up
in the run metadata of a field, and their reader, which reads them back
for the QA report as ``GainCalibrationSetXds``.

Gaintables are read through the shared ``create_gaintable_from_h5parm``.
The H5parm does not store the solution interval, so it is read back as
NaN; the QA report does not use it.
"""

from functools import lru_cache

import h5py
import numpy as np
import xarray as xr

from ...data_managers.gaintable import create_gaintable_from_h5parm
from ...data_managers.schema.calibration_set import GainCalibrationSetXds
from ..exceptions import RunMetadataError
from .field_metadata import FieldMetadata
from .manifest_schema import ManifestSchema

_SOLSET = "sol000"
# Soltabs of a gaintable; either may be absent, as read_h5parm_gains allows
_GAIN_SOLTABS = ("amplitude000", "phase000")


def read_gaintable(gaintable_paths: list[str]) -> GainCalibrationSetXds:
    """
    Read gaintables as one gaintable, concatenated along time, as the
    delay tables are, and ordered by time.

    Each gaintable is read once for consecutive queries of the same field.
    A single gaintable is returned as read, shared by every caller of the
    same path, so it must not be modified in place.

    Its solution interval is not stored in the H5parm, so
    ``CALPARAM_INTERVAL`` is NaN: the QA report does not use it.

    Parameters
    ----------
    gaintable_paths : list[str]
        H5parm gaintables, e.g. the final gaintable of a field.

    Returns
    -------
    GainCalibrationSetXds
        Gains and weights of dimensions (time, antenna_name, frequency,
        polarization), with ``XX XY YX YY`` polarizations.

    Raises
    ------
    RunMetadataError
        If there are no gaintables, one was not produced or cannot be
        read (it is corrupt, or has no gain soltab or no time, frequency
        or antenna axis), or they do not have the same stations and
        channels. Errors are not cached.
    """
    if not gaintable_paths:
        raise RunMetadataError("No gaintable recorded")

    gaintables = [_read_one(path) for path in gaintable_paths]

    # A single gaintable needs no concatenation, which would copy its
    # gains: the final gaintable of a field is a single one
    if len(gaintables) == 1:
        return gaintables[0]

    try:
        # join="exact": gaintables of different stations or channels are
        # a bad product, not padded
        gaintable = xr.concat(gaintables, dim="time", join="exact")
    except ValueError as error:
        raise RunMetadataError(
            f"Gaintables {gaintable_paths} do not have the same stations "
            f"and channels: {error}"
        ) from error

    return gaintable.sortby("time")


# maxsize=1: fields are reported one after another, and each of their
# queries reads the same final gaintable, so only the last read is reused.
# A bigger cache would keep the gains of every field in memory
@lru_cache(maxsize=1)
def _read_one(gaintable_path: str) -> GainCalibrationSetXds:
    """
    Read one gaintable, with a NaN solution interval.

    The last gaintable read is cached, so the queries of a field share its
    read. Errors are not cached.

    Parameters
    ----------
    gaintable_path : str
        Path of the H5parm gaintable.

    Returns
    -------
    GainCalibrationSetXds
        Gains and weights of the gaintable.

    Raises
    ------
    RunMetadataError
        If the gaintable was not produced, or cannot be read: it is
        corrupt, or has no gain soltab or no time, frequency or antenna
        axis.
    """
    try:
        interval = np.full(_count_times(gaintable_path), np.nan)
        return create_gaintable_from_h5parm(gaintable_path, interval)
    except FileNotFoundError as error:
        # The run metadata lists the products a run should write, so a
        # listed gaintable can be missing if its export failed
        raise RunMetadataError(
            f"Gaintable {gaintable_path} was not produced"
        ) from error
    except (OSError, KeyError, ValueError) as error:
        # A bad product, not a bug: a truncated or corrupt HDF5 file
        # (OSError), no solset or gain soltab (KeyError), or no time,
        # frequency or antenna axis (ValueError)
        raise RunMetadataError(
            f"Gaintable {gaintable_path} cannot be read: {error}"
        ) from error


def _count_times(gaintable_path: str) -> int:
    """
    Count the solution times of a gaintable, to size its interval.

    The times are read off the ``amplitude000`` soltab, or ``phase000`` if
    it is absent. A gaintable without a time axis counts one:
    create_gaintable_from_h5parm then raises, as it cannot restore the
    axis without the visibilities.

    Parameters
    ----------
    gaintable_path : str
        Path of the H5parm gaintable.

    Returns
    -------
    int
        Number of solution times, or 1 if the gaintable has no time axis.

    Raises
    ------
    FileNotFoundError
        If the gaintable does not exist.
    OSError
        If the gaintable is not a readable HDF5 file.
    KeyError
        If the gaintable has no solset or gain soltab.
    """
    with h5py.File(gaintable_path, "r") as h5f:
        solset = h5f[_SOLSET]
        soltab_name = next(
            (name for name in _GAIN_SOLTABS if name in solset), None
        )
        if soltab_name is None:
            raise KeyError(
                f"No {' or '.join(_GAIN_SOLTABS)} soltab in {_SOLSET}"
            )

        soltab = solset[soltab_name]
        return len(soltab["time"]) if "time" in soltab else 1


def get_final_gaintable_path(field_metadata: FieldMetadata) -> str:
    """
    Get the final gaintable path of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field.

    Returns
    -------
    str
        Path of the final gaintable of the field.

    Raises
    ------
    RunMetadataError
        If no final gaintable is recorded for the field.
    """
    gaintable = field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.FINAL_GAINTABLE
    )
    if not gaintable:
        raise RunMetadataError(
            f"No final gaintable recorded for field "
            f"{field_metadata.field_name}"
        )
    return gaintable


def get_bandpass_gaintable_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the bandpass gaintables of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one bandpass gaintable recorded
        per MS prefix.

    Returns
    -------
    list[str]
        Bandpass gaintable of each measurement set of the field. Empty if
        none is recorded: the reader raises for it.
    """
    gaintables = (
        field_metadata.get_data_for(ManifestSchema.BANDPASS.GAINTABLE) or {}
    )
    return list(gaintables.values())
