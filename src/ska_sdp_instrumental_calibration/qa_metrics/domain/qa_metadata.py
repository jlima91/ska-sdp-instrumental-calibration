import re
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

from astropy.time import Time

from ska_sdp_instrumental_calibration.xarray_processors.vis_flagging import (
    flagged_sample_counts,
    unsolvable_gain_counts,
)

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
        return cls(
            {
                str(station_id): str(name)
                for station_id, name in enumerate(
                    ms_qa_data[0]["station_names"]
                )
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
    def from_ms_qa_data(
        cls, ms_qa_data: list[dict], station_names: list[str]
    ) -> "PreFlagging":
        """
        Get the flagging of a field from the QA data of its measurement
        sets.

        Parameters
        ----------
        ms_qa_data : list[dict]
            QA data of each measurement set of the field, from
            get_ms_qa_data.
        station_names : list[str]
            MS station names, in station ID order.

        Returns
        -------
        PreFlagging
            Flagging of the field.
        """
        return cls(
            flagged_data=cls._percentage_per_station(
                flagged_sample_counts, ms_qa_data, station_names
            ),
            unsolvable_gains=cls._percentage_per_station(
                unsolvable_gain_counts, ms_qa_data, station_names
            ),
        )

    @staticmethod
    def _percentage_per_station(
        count_function: Callable,
        ms_qa_data: list[dict],
        station_names: list[str],
    ) -> dict[str, float]:
        """
        Sum the (flagged, total) counts over the measurement sets, then
        return the percentage keyed by station name.
        """
        flagged = 0
        total = 0
        for ms in ms_qa_data:
            ms_flagged, ms_total = count_function(
                ms["flagged_time_count"],
                ms["antenna1"],
                ms["antenna2"],
                ms["ntimes"],
                len(station_names),
            )
            flagged = flagged + ms_flagged
            total = total + ms_total

        percentages = flagged / total * 100
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
    source_name : str
        Field name.
    file_information : FileInformation
        Files of the field.
    observation_parameters : ObservationParameters
        Observation parameters.
    station_configuration : StationConfiguration
        Stations of the field.
    pre_flagging : PreFlagging
        Flagging before calibration.
    """

    source_name: str
    file_information: FileInformation
    observation_parameters: ObservationParameters
    station_configuration: StationConfiguration
