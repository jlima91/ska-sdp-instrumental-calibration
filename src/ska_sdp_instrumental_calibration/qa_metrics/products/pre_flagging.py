"""
Pre-flagging data: the counts of the flags already set in the visibilities
of a measurement set before calibration, kept in the ``derived/`` folder as
one JSON file per measurement set. Its layout is defined by the writer and
reader below, its only users.
"""

import json
import logging

import numpy as np
import xarray as xr

from ...xarray_processors.vis_flagging import (
    flagged_sample_counts,
    flagged_time_count,
    unsolvable_gain_counts,
)
from ..exceptions import RunMetadataError

logger = logging.getLogger(__name__)

# Keys of the per station count lists in a pre-flagging file
_STATION_COUNT_KEYS = ("flagged", "total", "unsolvable")


def export_pre_flagging(vis: xr.Dataset, pre_flagging_output_path: str):
    """
    Export the counts of the flags already set in the visibilities, before
    calibration, per station, to a JSON file.

    Counts rather than percentages are stored, so that the counts of the
    measurement sets of a field can be summed before dividing.

    Parameters
    ----------
    vis : Visibility
        Visibility dataset, with ``flags`` of dimensions (``time``,
        ``baselineid``, ``frequency``, ``polarisation``) and the
        ``antenna1`` and ``antenna2`` station index of each baseline.
    pre_flagging_output_path : str
        Path of the JSON file to write, with keys:

        ``flagged``, ``total``
            Flagged and total visibility sample count per station, from
            :func:`flagged_sample_counts`.
        ``unsolvable``
            Count of channels per station whose gain cannot be solved, as
            all their visibilities are flagged, from
            :func:`unsolvable_gain_counts`.
        ``nchannels``
            Number of gain solutions of a station: one per channel.

        Per station lists are in configuration order.
    """
    # Computes the lazy flags: a (baselineid, frequency) count array
    flagged_count = flagged_time_count(vis).values
    station_args = (
        flagged_count,
        vis.antenna1.values,
        vis.antenna2.values,
        vis.time.size,
        vis.configuration.id.size,
    )

    flagged, total = flagged_sample_counts(*station_args)
    unsolvable, nchannels = unsolvable_gain_counts(*station_args)

    pre_flagging = {
        "flagged": flagged.astype(int).tolist(),
        "total": total.astype(int).tolist(),
        "unsolvable": unsolvable.astype(int).tolist(),
        "nchannels": int(nchannels),
    }

    logger.info("Writing pre-flagging data to %s", pre_flagging_output_path)
    with open(pre_flagging_output_path, "w", encoding="utf-8") as output_file:
        json.dump(pre_flagging, output_file)


def read_pre_flagging(
    pre_flagging_paths: list[str], nstations: int
) -> list[dict]:
    """
    Read the pre-flagging files of the measurement sets of a field, written
    by :func:`export_pre_flagging`.

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
        :func:`export_pre_flagging`.
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
