import re
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy.time import Time

QA_METADATA_FILE_NAME = "qa-metadata-data.json"


@dataclass
class FileInformation:
    """
    Files of a field.

    Parameters
    ----------
    input_ms : list[str]
        Paths of the measurement sets of the field.
    output_path : str
        QA directory of the run: the SDM logs directory or the output
        directory.
    """

    input_ms: list[str]
    output_path: str

    @classmethod
    def from_ms_qa_data(
        cls, ms_qa_data: list[dict], output_path: str
    ) -> "FileInformation":
        """
        Get the files of a field from the QA data of its measurement sets.

        Parameters
        ----------
        ms_qa_data : list[dict]
            QA data of each measurement set of the field, from
            get_ms_qa_data.
        output_path : str
            QA directory of the run.

        Returns
        -------
        FileInformation
            Files of the field.
        """
        return cls([ms["input_ms"] for ms in ms_qa_data], output_path)


@dataclass
class ObservationParameters:
    """
    Observation parameters of a field.

    Parameters
    ----------
    start_time : float
        Start of the observation, in MJD seconds as stored in the MS.
    observation_length : float
        Length of the observation, in seconds.
    integration_time : float
        Integration time, in seconds.
    channels : int or None
        Number of frequency channels, or None if not recorded; the report
        shows it as unknown.
    """

    start_time: float
    observation_length: float
    integration_time: float
    channels: int | None

    def start_time_utc(self) -> str:
        """
        Start time as a UTC date and time string.

        Returns
        -------
        str
            UTC time, e.g. ``2026-02-12 10:00:00 UTC``.
        """
        utc = Time(
            self.start_time / 86400.0, format="mjd", scale="utc", precision=0
        )
        return f"{utc.iso} UTC"


@dataclass
class StationConfiguration:
    """
    Stations of a field.

    Parameters
    ----------
    station_names : dict[str, str]
        Station names, as in the MS, keyed by station ID.
    """

    station_names: dict[str, str]

    @classmethod
    def from_ms_qa_data(cls, ms_qa_data: list[dict]) -> "StationConfiguration":
        """
        Get the stations of a field from the QA data of its measurement
        sets.

        Parameters
        ----------
        ms_qa_data : list[dict]
            QA data of each measurement set of the field, from
            get_ms_qa_data.

        Returns
        -------
        StationConfiguration
            Stations of the field.
        """
        # Measurement sets of a field share stations, so they are taken
        # from the first one
        return cls.from_station_names(ms_qa_data[0]["station_names"])

    @classmethod
    def from_station_names(cls, station_names) -> "StationConfiguration":
        """
        Get the stations of a field from their names, in configuration
        order.

        Parameters
        ----------
        station_names : Iterable
            MS station names, e.g. the antenna axis of a gaintable. Their
            position is their station ID.

        Returns
        -------
        StationConfiguration
            Stations of the field.
        """
        return cls(
            {
                str(station_id): str(name)
                for station_id, name in enumerate(station_names)
            }
        )

    @cached_property
    def short_station_names(self) -> dict[str, str]:
        """
        Smaller station names, e.g. S8-1, for display. Computed once.

        Returns
        -------
        dict[str, str]
            Smaller station name keyed by MS station name. The MS name is
            kept if it has no smaller name.
        """
        short_station_names = {}
        for name in self.station_names.values():
            match = re.search(r"_([^_()]+)\)$", name)
            short_station_names[name] = match.group(1) if match else name

        return short_station_names


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

    @classmethod
    def from_counts(
        cls, counts: list[dict], station_names: list[str]
    ) -> "PreFlagging":
        """
        Get the flagging of a field from the pre-flagging counts of its
        measurement sets, as read by ``products.read_pre_flagging``.

        The counts of the measurement sets are summed before dividing, so
        each measurement set weighs by its number of samples.

        Parameters
        ----------
        counts : list[dict]
            Counts of each measurement set of the field: ``flagged``,
            ``total`` and ``unsolvable`` with one count per station, and
            ``nchannels``.
        station_names : list[str]
            MS station names, in the configuration order of the counts.

        Returns
        -------
        PreFlagging
            Flagging of the field.
        """
        flagged = np.sum(
            [ms_counts["flagged"] for ms_counts in counts], axis=0
        )
        total = np.sum([ms_counts["total"] for ms_counts in counts], axis=0)
        unsolvable = np.sum(
            [ms_counts["unsolvable"] for ms_counts in counts], axis=0
        )
        nchannels = sum(ms_counts["nchannels"] for ms_counts in counts)

        return cls(
            flagged_data=cls._percentage_per_station(
                flagged, total, station_names
            ),
            unsolvable_gains=cls._percentage_per_station(
                unsolvable, nchannels, station_names
            ),
        )

    @staticmethod
    def _percentage_per_station(
        count: np.ndarray,
        total: np.ndarray | int,
        station_names: list[str],
    ) -> dict[str, float]:
        """
        Percentage of count over total, keyed by station name.
        """
        percentages = count / total * 100
        return {
            name: float(percentage)
            for name, percentage in zip(station_names, percentages)
        }


@dataclass
class QAMetadata:
    """
    QA metadata of the measurement sets of one field, composed of one
    object per section.

    Parameters
    ----------
    file_information : FileInformation
        Files of the field.
    observation_parameters : ObservationParameters
        Observation parameters.
    station_configuration : StationConfiguration
        Stations of the field.
    pre_flagging : PreFlagging
        Flagging before calibration.
    """

    file_information: FileInformation
    observation_parameters: ObservationParameters
    station_configuration: StationConfiguration
