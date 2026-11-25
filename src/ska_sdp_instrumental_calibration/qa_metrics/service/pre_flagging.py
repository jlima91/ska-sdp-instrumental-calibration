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

from ..domain.flagging_statistics import PreFlagging
from ..exceptions import RunMetadataError
from ..repository.field_metadata import FieldMetadata
from ..repository.manifest_schema import ManifestSchema

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


def flagged_time_count(vis: xr.Dataset) -> xr.DataArray:
    """
    Number of flagged time steps per baseline and channel.

    A sample is flagged where the MS FLAG column is set, and the whole
    sample is flagged if any of its polarisations is.

    Parameters
    ----------
    vis: Visibility
        Visibility dataset.

    Returns
    -------
    xr.DataArray
        Lazy flagged time step count, with dimensions
        (``baselineid``, ``frequency``).
    """
    is_flagged = vis.flags.any(dim="polarisation")

    return is_flagged.sum(dim="time")


def _cross_correlations(
    flagged_count: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only the cross-correlation baselines.

    """
    is_cross = antenna1 != antenna2
    return flagged_count[is_cross], antenna1[is_cross], antenna2[is_cross]


def flagged_sample_counts(
    flagged_count: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ntimes: int,
    nstations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Number of flagged and of all visibility samples per station, over the
    cross-correlations of one visibility.

    Parameters
    ----------
    flagged_count: np.ndarray
        Flagged time step count per baseline and channel.
    antenna1: np.ndarray
        First station index of each baseline.
    antenna2: np.ndarray
        Second station index of each baseline.
    ntimes: int
        Number of time steps.
    nstations: int
        Number of stations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flagged and total sample count per station.
    """
    flagged_count, antenna1, antenna2 = _cross_correlations(
        flagged_count, antenna1, antenna2
    )
    flagged_per_baseline = flagged_count.sum(axis=1)
    samples_per_baseline = ntimes * flagged_count.shape[1]

    # np.bincount(idx, weights=w)[k] sums w over all baselines where
    # idx == k. Each baseline counts for both of its stations, hence
    # antenna1 + antenna2.
    flagged = np.bincount(
        antenna1, weights=flagged_per_baseline, minlength=nstations
    ) + np.bincount(
        antenna2, weights=flagged_per_baseline, minlength=nstations
    )
    total = (
        np.bincount(antenna1, minlength=nstations)
        + np.bincount(antenna2, minlength=nstations)
    ) * samples_per_baseline

    return flagged, total


def unsolvable_gain_counts(
    flagged_count: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ntimes: int,
    nstations: int,
) -> tuple[np.ndarray, int]:
    """
    Number of gain solutions per station that cannot be solved, because
    all visibilities of the station in that channel are flagged, and of
    all gain solutions, for one visibility.

    This counts in the same unit as the flagged gains of a gaintable: one
    solution per station and channel, as bandpass solves one solution over
    all time steps of a visibility.

    Parameters
    ----------
    flagged_count: np.ndarray
        Flagged time step count per baseline and channel.
    antenna1: np.ndarray
        First station index of each baseline.
    antenna2: np.ndarray
        Second station index of each baseline.
    ntimes: int
        Number of time steps.
    nstations: int
        Number of stations.

    Returns
    -------
    tuple[np.ndarray, int]
        Unsolvable gain count per station, and the number of gain
        solutions of a station.
    """
    flagged_count, antenna1, antenna2 = _cross_correlations(
        flagged_count, antenna1, antenna2
    )
    nchannels = flagged_count.shape[1]
    # A baseline channel has data when not all its time steps are flagged
    has_data = (flagged_count < ntimes).astype(int)

    # Number of baselines with data per station and channel, in one pass
    # over the baselines: np.add.at adds each baseline row to both of its
    # stations, like np.bincount for the sample counts
    baselines_with_data = np.zeros((nstations, nchannels), dtype=int)
    np.add.at(baselines_with_data, antenna1, has_data)
    np.add.at(baselines_with_data, antenna2, has_data)

    # A channel is unsolvable when no baseline of the station has data
    unsolvable = (baselines_with_data == 0).sum(axis=1)

    return unsolvable, nchannels


def _percentage_per_station(
    count: np.ndarray,
    total: np.ndarray | int,
    station_names: list[str],
) -> dict[str, float]:
    """
    Compute the percentage of a count over its total, per station.

    Parameters
    ----------
    count : np.ndarray
        Count per station, in the order of ``station_names``.
    total : np.ndarray or int
        Total per station, or one total shared by all stations.
    station_names : list[str]
        MS station names, in the order of the counts.

    Returns
    -------
    dict[str, float]
        Percentage keyed by MS station name. NaN for a station whose
        total is 0.
    """
    percentages = count / total * 100
    return {
        name: float(percentage)
        for name, percentage in zip(station_names, percentages)
    }


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


def pre_flags_from_counts(
    counts: list[dict], station_names: list[str]
) -> PreFlagging:
    """
    Get the flagging of a field from the pre-flagging counts of its
    measurement sets, as read by ``products.read_pre_flagging``.

    The counts of the measurement sets are summed before dividing, so
    each measurement set weighs by its number of samples.

    Parameters
    ----------
    counts : list[dict]
        Counts of each measurement set of the field, at least one:
        ``flagged``, ``total`` and ``unsolvable`` with one count per
        station, and ``nchannels``.
    station_names : list[str]
        MS station names, in the configuration order of the counts.

    Returns
    -------
    PreFlagging
        Flagging of the field.
    """
    flagged = np.sum([ms_counts["flagged"] for ms_counts in counts], axis=0)
    total = np.sum([ms_counts["total"] for ms_counts in counts], axis=0)
    unsolvable = np.sum(
        [ms_counts["unsolvable"] for ms_counts in counts], axis=0
    )
    nchannels = sum(ms_counts["nchannels"] for ms_counts in counts)

    return PreFlagging(
        flagged_data=_percentage_per_station(flagged, total, station_names),
        unsolvable_gains=_percentage_per_station(
            unsolvable, nchannels, station_names
        ),
    )


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
