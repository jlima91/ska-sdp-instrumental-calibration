"""
Pre-flagging data files, one JSON file per measurement set in the
``derived/`` folder: their writer, their paths, looked up in the run
metadata of a field, and their reader. Their layout is defined here, by
the writer and reader, its only users. The counts they hold are computed
and exported by :func:`..service.pre_flagging.export_pre_flagging`.
"""

import json
import logging

import numpy as np

from ..exceptions import RunMetadataError
from .field_metadata import FieldMetadata
from .manifest_schema import ManifestSchema

logger = logging.getLogger(__name__)

# Keys of the per station count lists in a pre-flagging file
_STATION_COUNT_KEYS = ("flagged", "total", "unsolvable")


def write_pre_flagging(counts: dict, pre_flagging_output_path: str):
    """
    Write the pre-flagging counts of a measurement set to a JSON file.

    Parameters
    ----------
    counts : dict
        Counts from :func:`..service.pre_flagging.count_pre_flagging`:
        ``flagged``, ``total`` and ``unsolvable`` with one count per
        station, in configuration order, and ``nchannels``.
    pre_flagging_output_path : str
        Path of the JSON file to write.
    """
    logger.info("Writing pre-flagging data to %s", pre_flagging_output_path)
    with open(pre_flagging_output_path, "w", encoding="utf-8") as output_file:
        json.dump(counts, output_file)


def get_pre_flagging_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the pre-flagging data files of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one pre-flagging data file
        recorded per MS prefix.

    Returns
    -------
    list[str]
        Pre-flagging data file of each measurement set of the field. Empty
        if none is recorded: the reader raises for it.
    """
    pre_flagging = (
        field_metadata.get_data_for(ManifestSchema.FLAGGING.PRE_FLAGGING) or {}
    )
    return list(pre_flagging.values())


def read_pre_flagging(
    pre_flagging_paths: list[str], nstations: int
) -> list[dict]:
    """
    Read the pre-flagging files of the measurement sets of a field, written
    by :func:`write_pre_flagging`.

    Parameters
    ----------
    pre_flagging_paths : list[str]
        Pre-flagging JSON file of each measurement set of the field.
    nstations : int
        Number of stations each file must have counts for.

    Returns
    -------
    list[dict]
        Counts of each file: ``flagged``, ``total`` and ``unsolvable`` as
        arrays of one count per station, in configuration order, and
        ``nchannels`` as an int.

    Raises
    ------
    RunMetadataError
        If there are no files, a file was not produced or cannot be read,
        or its counts are missing or not one per station.
    """
    if not pre_flagging_paths:
        raise RunMetadataError("No pre-flagging data recorded")

    return [_read_counts(path, nstations) for path in pre_flagging_paths]


def _read_counts(path: str, nstations: int) -> dict:
    """
    Read the counts of one pre-flagging file, checking there is one per
    station.

    Parameters
    ----------
    path : str
        Path of the pre-flagging JSON file, written by
        :func:`write_pre_flagging`.
    nstations : int
        Number of stations the file must have counts for.

    Returns
    -------
    dict
        ``flagged``, ``total`` and ``unsolvable`` as arrays of one count
        per station, and ``nchannels`` as an int.

    Raises
    ------
    RunMetadataError
        If the file was not produced or cannot be read, or its counts are
        missing or not one per station.
    """
    try:
        with open(path, encoding="utf-8") as pre_flagging_file:
            counts = json.load(pre_flagging_file)
    except FileNotFoundError as error:
        # The run metadata lists the files a run should write, so a listed
        # file can be missing if its export failed
        raise RunMetadataError(
            f"Pre-flagging data {path} was not produced"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        # A bad product, not a bug: e.g. a file left half written by a
        # failed export
        raise RunMetadataError(
            f"Pre-flagging data {path} cannot be read: {error}"
        ) from error

    missing = [
        key for key in (*_STATION_COUNT_KEYS, "nchannels") if key not in counts
    ]
    if missing:
        raise RunMetadataError(
            f"Pre-flagging data {path} has no {', '.join(missing)} counts"
        )

    for key in _STATION_COUNT_KEYS:
        if len(counts[key]) != nstations:
            raise RunMetadataError(
                f"Pre-flagging data {path} has {len(counts[key])} "
                f"{key} counts for {nstations} stations"
            )

    return {
        **{key: np.asarray(counts[key]) for key in _STATION_COUNT_KEYS},
        "nchannels": int(counts["nchannels"]),
    }
