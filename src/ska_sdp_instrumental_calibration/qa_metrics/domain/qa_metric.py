import os
from dataclasses import dataclass

from ska_sdp_instrumental_calibration.data_managers.h5parm import (
    read_h5parm_gains,
)

from .qa_metadata import PreFlagging, QAMetadata


def _check_gaintable_exists(final_gaintable_path: str):
    """Raise FileNotFoundError if the final gaintable is missing"""
    if not os.path.exists(final_gaintable_path):
        raise FileNotFoundError(
            f"No final gaintable found at {final_gaintable_path}"
        )


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
    def read_from_gaintable(
        cls, final_gaintable_path: str, pre_flagging: PreFlagging
    ) -> "FlaggingStatistics":
        """
        Get the flagging statistics of a field from its final gaintable.

        Parameters
        ----------
        final_gaintable_path : str
            Path of the final gaintable (h5parm) of the field.
        pre_flagging : PreFlagging
            Flagging of the field before calibration, from the QA metadata.

        Returns
        -------
        FlaggingStatistics
            Flagging statistics of the field.

        Raises
        ------
        FileNotFoundError
            If the final gaintable is missing.
        """
        _check_gaintable_exists(final_gaintable_path)
        return cls(
            flagged_before=pre_flagging.unsolvable_gains,
            flagged_after=cls._gaintable_flagged_percentage(
                final_gaintable_path
            ),
        )

    @staticmethod
    def _gaintable_flagged_percentage(h5parm_path: str) -> dict[str, float]:
        """
        Percentage of flagged gain solutions per station in an h5parm.
        A gain is flagged where its weight is 0. Only the parallel hand
        gains (XX, YY) solved by bandpass are counted.
        """
        _, weight, _, soltab = read_h5parm_gains(h5parm_path)

        is_flagged = weight[..., [0, 1], [0, 1]] == 0
        flagged = is_flagged.sum(axis=(0, 2, 3))
        total = is_flagged[:, 0, ...].size

        return {
            str(name): float(percentage)
            for name, percentage in zip(soltab.antenna, flagged / total * 100)
        }

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


def _average(percentages) -> float | None:
    """Average of the percentages, None if there are none"""
    percentages = list(percentages)
    return sum(percentages) / len(percentages) if percentages else None


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
        """Metric values per station. Not implemented yet."""
        pass  # pylint: disable=unnecessary-pass


@dataclass
class CableDelays:
    """
    X and Y cable delays of a field, per station.

    Parameters
    ----------
    delays_per_station : dict[str, dict[str, float]], optional
        Delays keyed by column key (see CABLE_DELAYS_COLUMNS), per MS
        station name. None while not calculated.
    """

    delays_per_station: dict[str, dict[str, float]] | None

    @classmethod
    def from_qa_metadata(cls, metadata: QAMetadata) -> "CableDelays":
        """
        Get the cable delays of a field.

        Parameters
        ----------
        metadata : QAMetadata
            QA metadata of the field.

        Returns
        -------
        CableDelays
            Cable delays of the field.
        """
        return cls(cls._calculate_delays(metadata))

    def values_per_station(self) -> dict[str, dict[str, float]] | None:
        """
        Delays per station.

        Returns
        -------
        dict[str, dict[str, float]] or None
            Delays keyed by column key, per MS station name.
        """
        return self.delays_per_station

    @staticmethod
    def _calculate_delays(
        metadata: QAMetadata,
    ) -> dict[str, dict[str, float]] | None:
        """Delays per station. Not implemented yet."""
        pass  # pylint: disable=unnecessary-pass
