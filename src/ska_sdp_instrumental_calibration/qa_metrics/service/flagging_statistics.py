"""
Flagging statistics of a field: the flagged percentage of each station
after calibration, from its final gaintable.
"""

from ...data_managers.schema.calibration_set import GainCalibrationSetXds
from ..domain.flagging_statistics import FlaggingStatistics


def get_flagged_from_gaintable(
    gaintable: GainCalibrationSetXds, flagged_before: dict[str, float]
) -> FlaggingStatistics:
    """
    Get the flagging statistics of a field from its final gaintable.

    A gain is flagged where its weight is 0. Only the parallel hand
    gains (XX, YY) solved by bandpass are counted.

    Parameters
    ----------
    gaintable : GainCalibrationSetXds
        Final gaintable of the field, with ``CALPARAM_WEIGHT`` of
        dimensions (time, antenna_name, frequency, polarization).
    flagged_before : dict[str, float]
        Percentage of gains that cannot be solved before calibration,
        keyed by MS station name.

    Returns
    -------
    FlaggingStatistics
        Flagging statistics of the field.
    """
    is_flagged = gaintable.CALPARAM_WEIGHT.sel(polarization=["XX", "YY"]) == 0
    # Percentage of the gains of each station: all its times,
    # channels and parallel hands
    flagged_percentage = (
        is_flagged.mean(dim=("time", "frequency", "polarization")) * 100
    )
    percentage_flagged = {
        str(name): float(percentage)
        for name, percentage in zip(
            gaintable.antenna_name.values, flagged_percentage.values
        )
    }

    return FlaggingStatistics(
        flagged_before=_average(flagged_before.values()),
        flagged_after=_average(percentage_flagged.values()),
        stats_per_station=compute_flags_per_station(
            flagged_before, percentage_flagged
        ),
    )


def compute_flags_per_station(
    flagged_before, flagged_after
) -> dict[str, dict[str, float | None]]:
    """
    Flagged percentage before, after and additional (after - before)
    per station.

    Returns
    -------
    dict[str, dict[str, float | None]]
        ``flagged_before``, ``flagged_after`` and ``flagged_additional``
        keyed by MS station name. A percentage is None where it is not
        available.
    """
    values = {}
    for station_name in flagged_before.keys() | flagged_after:
        before = flagged_before.get(station_name)
        after = flagged_after.get(station_name)
        values[station_name] = {
            "flagged_before": before,
            "flagged_after": after,
            "flagged_additional": (
                after - before
                if before is not None and after is not None
                else None
            ),
        }

    return values


def _average(percentages) -> float | None:
    """
    Compute the average of percentages.

    Parameters
    ----------
    percentages : Iterable[float]
        Percentages to average.

    Returns
    -------
    float or None
        Average percentage, or None if there are no percentages.
    """
    percentages = list(percentages)
    return sum(percentages) / len(percentages) if percentages else None
