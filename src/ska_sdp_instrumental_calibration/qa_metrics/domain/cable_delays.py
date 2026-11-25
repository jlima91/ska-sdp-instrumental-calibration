"""
Metrics of a field for the QA report, computed from its read products:
flagging statistics from its final gaintable, bandpass fit metrics, and
cable delays from its delay tables.
"""

from dataclasses import dataclass


@dataclass
class CableDelays:
    """
    X and Y cable delays of a field, per station.

    Parameters
    ----------
    delays_per_station : dict[str, dict[str, float]]
        ``x_delay`` and ``y_delay`` in nanoseconds (see
        CABLE_DELAYS_COLUMNS), per MS station name.
    """

    delays_per_station: dict[str, dict[str, float]]

    @property
    def values_per_station(self) -> dict[str, dict[str, float]]:
        """
        Delays per station.

        Returns
        -------
        dict[str, dict[str, float]]
            ``x_delay`` and ``y_delay`` in nanoseconds, per MS station
            name.
        """
        return self.delays_per_station
