"""
Flagging of a field for the QA report: its pre-flagging, computed from
the flag counts of its measurement sets before calibration, and its
flagging statistics, before and after calibration.
"""

from dataclasses import dataclass


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


@dataclass
class PreFlagging:
    """
    Flagging of a field before calibration, in percentage per station.

    Parameters
    ----------
    flagged_data : dict[str, float]
        Flagged percentage of the visibilities, keyed by MS station name.
    unsolvable_gains : dict[str, float]
        Percentage of gains that cannot be solved because all their
        visibilities are flagged, keyed by MS station name.
    """

    flagged_data: dict[str, float]
    unsolvable_gains: dict[str, float]


@dataclass
class FlaggingStatistics:
    """
    Flagged gain solutions of a field, in percentage per station, before
    and after calibration.

    Parameters
    ----------
    flagged_before : dict[str, float]
        Percentage of gains that cannot be solved because all their
        visibilities are flagged, keyed by MS station name.
    flagged_after : dict[str, float]
        Percentage of gains flagged in the final gaintable, keyed by MS
        station name.
    """

    flagged_before: dict[str, float]
    flagged_after: dict[str, float]

    @property
    def values_per_station(self) -> dict[str, dict[str, float | None]]:
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
        for station_name in self.flagged_before.keys() | self.flagged_after:
            before = self.flagged_before.get(station_name)
            after = self.flagged_after.get(station_name)
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

    @property
    def totals(self) -> dict[str, float | None]:
        """
        Flagged percentage over all stations, before and after calibration.

        Returns
        -------
        dict[str, float | None]
            ``flagged_before`` and ``flagged_after`` average percentage,
            None where no station has the percentage.
        """
        return {
            "flagged_before": _average(self.flagged_before.values()),
            "flagged_after": _average(self.flagged_after.values()),
        }
