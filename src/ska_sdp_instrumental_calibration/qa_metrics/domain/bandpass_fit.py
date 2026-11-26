"""
Bandpass fit metrics of a field for the QA report: the phase and gain
amplitude fit quality of each station, computed from its bandpass
gaintables.
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
        get_bandpass_fit_metrics_columns), per MS station name: NaN where
        a station has too few channels to fit. None while not calculated.
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
