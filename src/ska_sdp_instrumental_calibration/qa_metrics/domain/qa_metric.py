"""
Metrics of a field for the QA report, computed from its read products:
flagging statistics from its final gaintable, bandpass fit metrics, and
cable delays from its delay tables.
"""

import os
from dataclasses import dataclass

import xarray as xr

from ...data_managers.schema.calibration_set import GainCalibrationSetXds


def _check_gaintable_exists(final_gaintable_path: str):
    """
    Check that the final gaintable exists.

    Only used by BandpassFitMetrics, which still reads its gaintable by
    path; to be removed once it is built from a read gaintable.

    Parameters
    ----------
    final_gaintable_path : str
        Path of the final gaintable.

    Raises
    ------
    FileNotFoundError
        If the final gaintable is missing.
    """
    if not os.path.exists(final_gaintable_path):
        raise FileNotFoundError(
            f"No final gaintable found at {final_gaintable_path}"
        )


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

    @classmethod
    def from_gaintable(
        cls, gaintable: GainCalibrationSetXds, flagged_before: dict[str, float]
    ) -> "FlaggingStatistics":
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
        is_flagged = (
            gaintable.CALPARAM_WEIGHT.sel(polarization=["XX", "YY"]) == 0
        )
        # Percentage of the gains of each station: all its times,
        # channels and parallel hands
        flagged_percentage = (
            is_flagged.mean(dim=("time", "frequency", "polarization")) * 100
        )

        return cls(
            flagged_before=flagged_before,
            flagged_after={
                str(name): float(percentage)
                for name, percentage in zip(
                    gaintable.antenna_name.values, flagged_percentage.values
                )
            },
        )

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

    @classmethod
    def read_from_gaintable(
        cls, final_gaintable_path: str
    ) -> "BandpassFitMetrics":
        """
        Get the bandpass fit metrics of a field from its final gaintable.

        Parameters
        ----------
        final_gaintable_path : str
            Path of the final gaintable (h5parm) of the field.

        Returns
        -------
        BandpassFitMetrics
            Bandpass fit metrics of the field.

        Raises
        ------
        FileNotFoundError
            If the final gaintable is missing.
        """
        _check_gaintable_exists(final_gaintable_path)
        return cls(cls._calculate_metrics(final_gaintable_path))

    def values_per_station(self) -> dict[str, dict[str, float]] | None:
        """
        Metric values per station.

        Returns
        -------
        dict[str, dict[str, float]] or None
            Metric values keyed by column key, per MS station name.
        """
        return self.metrics_per_station

    @staticmethod
    def _calculate_metrics(
        final_gaintable_path: str,
    ) -> dict[str, dict[str, float]] | None:
        """
        Compute the metric values per station. Not implemented yet.

        Parameters
        ----------
        final_gaintable_path : str
            Path of the final gaintable of the field.

        Returns
        -------
        dict[str, dict[str, float]] or None
            Metric values keyed by column key, per MS station name. None
            until implemented.
        """
        pass  # pylint: disable=unnecessary-pass


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

    @classmethod
    def from_delaytable(cls, delaytable: xr.Dataset) -> "CableDelays":
        """
        Get the cable delays of a field from its delay table.

        A cable delay does not change over an observation, so each
        station gets the median of its delays over time: unlike the mean,
        it is barely moved by the delays of one badly solved scan. Delays
        that were not solved (NaN) are left out.

        Parameters
        ----------
        delaytable : DelayTable
            Delay table of the field, with ``delay`` in seconds of
            dimensions (time, antenna_name, pol) and ``XX`` and ``YY``
            polarisations.

        Returns
        -------
        CableDelays
            Cable delays of the field.
        """
        # Nanoseconds, as in CABLE_DELAYS_COLUMNS
        median_delays = delaytable.delay.median(dim="time", skipna=True) * 1e9
        x_delays = median_delays.sel(pol="XX").values
        y_delays = median_delays.sel(pol="YY").values

        return cls(
            {
                str(name): {"x_delay": float(x), "y_delay": float(y)}
                for name, x, y in zip(
                    delaytable.antenna_name.values, x_delays, y_delays
                )
            }
        )

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
