"""
Metadata of a field for the QA report: its files, observation parameters
and stations, and its flagging before calibration.
"""

from dataclasses import dataclass

from astropy.time import Time


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


@dataclass
class QAMetadata:
    """
    QA metadata of the measurement sets of one field, composed of one
    object per part of the metadata summary.

    Parameters
    ----------
    file_information : FileInformation
        Files of the field.
    observation_parameters : ObservationParameters
        Observation parameters.
    station_configuration : StationConfiguration
        Stations of the field.
    """

    file_information: FileInformation
    observation_parameters: ObservationParameters
    station_configuration: StationConfiguration
