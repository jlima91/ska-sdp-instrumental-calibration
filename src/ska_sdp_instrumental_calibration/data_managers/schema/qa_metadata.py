import json
import re
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Callable

import numpy as np
from astropy.time import Time

from ska_sdp_instrumental_calibration.exceptions import QAMetadataError
from ska_sdp_instrumental_calibration.xarray_processors.vis_flagging import (
    flagged_sample_counts,
    unsolvable_gain_counts,
)


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
    channels : int
        Number of frequency channels.
    """

    start_time: float
    observation_length: float
    integration_time: float
    channels: int

    @classmethod
    def from_ms_qa_data(
        cls, ms_qa_data: list[dict]
    ) -> "ObservationParameters":
        """
        Get the observation parameters of a field from the QA data of its
        measurement sets, joined along time.

        Parameters
        ----------
        ms_qa_data : list[dict]
            QA data of each measurement set of the field, from
            get_ms_qa_data.

        Returns
        -------
        ObservationParameters
            Observation parameters of the field.
        """
        # Measurement sets of a field share channels and integration time,
        # so they are taken from the first one
        first_ms = ms_qa_data[0]
        integration_time = np.median(np.diff(first_ms["timestamps"]))

        start_time = (
            min(ms["timestamps"].min() for ms in ms_qa_data)
            - integration_time / 2
        )
        end_time = (
            max(ms["timestamps"].max() for ms in ms_qa_data)
            + integration_time / 2
        )

        return cls(
            start_time=float(start_time),
            observation_length=float(end_time - start_time),
            integration_time=float(integration_time),
            channels=int(first_ms["channels"]),
        )

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
    pre_flagging: PreFlagging

    @classmethod
    def from_ms_qa_data(
        cls, source_name: str, ms_qa_data: list[dict], output_path: str
    ) -> "QAMetadata":
        """
        Get the QA metadata of a field from the QA data of its measurement
        sets.

        Parameters
        ----------
        source_name : str
            Field name.
        ms_qa_data : list[dict]
            QA data of each measurement set of the field, from
            get_ms_qa_data.
        output_path : str
            QA directory of the run.

        Returns
        -------
        QAMetadata
            QA metadata of the field.
        """
        station_configuration = StationConfiguration.from_ms_qa_data(
            ms_qa_data
        )
        return cls(
            source_name=source_name,
            file_information=FileInformation.from_ms_qa_data(
                ms_qa_data, output_path
            ),
            observation_parameters=ObservationParameters.from_ms_qa_data(
                ms_qa_data
            ),
            station_configuration=station_configuration,
            pre_flagging=PreFlagging.from_ms_qa_data(
                ms_qa_data,
                list(station_configuration.station_names.values()),
            ),
        )

    def write_to_json(self, path: str):
        """
        Write the QA metadata to a JSON file, one object per section.

        Parameters
        ----------
        path : str
            Path of the JSON file.
        """
        with open(path, "w", encoding="utf-8") as metadata_file:
            json.dump(asdict(self), metadata_file, indent=4)

    @classmethod
    def read_from_json(cls, path: str) -> "QAMetadata":
        """
        Read the QA metadata from a JSON file.

        Parameters
        ----------
        path : str
            Path of the JSON file.

        Returns
        -------
        QAMetadata
            The QA metadata.

        Raises
        ------
        QAMetadataError
            If the file is malformed or incomplete.
        """
        try:
            with open(path, encoding="utf-8") as metadata_file:
                qa_metadata = json.load(metadata_file)

            return cls(
                source_name=qa_metadata["source_name"],
                file_information=FileInformation(
                    **qa_metadata["file_information"]
                ),
                observation_parameters=ObservationParameters(
                    **qa_metadata["observation_parameters"]
                ),
                station_configuration=StationConfiguration(
                    **qa_metadata["station_configuration"]
                ),
                pre_flagging=PreFlagging(**qa_metadata["pre_flagging"]),
            )
        except (ValueError, KeyError, TypeError) as error:
            raise QAMetadataError(
                f"Invalid QA metadata {path}: {error}"
            ) from error
