"""
Metrics of a field for the QA report, computed from its read products:
flagging statistics from its final gaintable, bandpass fit metrics, and
cable delays from its delay tables.
"""

from dataclasses import dataclass


@dataclass
class BandpassFitMetrics:
    """
    Fit metrics of the bandpass gains of a field, per station.

    Parameters
    ----------
    metrics_per_station : dict[str, dict[str, float]], optional
        Metric values keyed by column key (see
        BANDPASS_FIT_METRICS_COLUMNS), per MS station name. None while
        not calculated.
    """

    metrics_per_station: dict[str, dict[str, float]] | None

    @property
    def values_per_station(self) -> dict[str, dict[str, float]] | None:
        """
        Metric values per station.

        Returns
        -------
        dict[str, dict[str, float]] or None
            Metric values keyed by column key, per MS station name.
        """
        return self.metrics_per_station
