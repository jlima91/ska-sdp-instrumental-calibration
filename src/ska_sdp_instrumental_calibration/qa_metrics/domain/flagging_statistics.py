"""
Flagging of a field for the QA report: its pre-flagging, computed from
the flag counts of its measurement sets before calibration, and its
flagging statistics, before and after calibration.
"""

from dataclasses import dataclass


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
    stats_per_station: dict[str, dict[str, float]] | None

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

        return self.stats_per_station

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
            "flagged_before": self.flagged_before,
            "flagged_after": self.flagged_after,
        }
