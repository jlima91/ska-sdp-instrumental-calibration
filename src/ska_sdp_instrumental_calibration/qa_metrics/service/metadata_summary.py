"""
Observation parameters of a field, computed from the timestamps of its
measurement sets. The timestamps, channels, stations and measurement sets
are looked up by :mod:`..repository.metadata`.
"""

import numpy as np

from ..domain.metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from ..exceptions import RunMetadataError
from ..repository import metadata as repository
from ..repository.field_metadata import FieldMetadata


def get_observation_parameters(
    timestamps: list[list[float]], channels: int | None
) -> ObservationParameters:
    """
    Get the observation parameters of a field from the timestamps of its
    measurement sets.

    The integration time is the median spacing of the timestamps of the
    first measurement set. The observation spans the timestamps of all
    measurement sets, which mark the centre of each integration.

    Parameters
    ----------
    timestamps : list[list[float]]
        Timestamps of each measurement set of the field, at least one, in
        MJD seconds, in any order. See
        :func:`..repository.metadata.get_timestamps`.
    channels : int or None
        Number of frequency channels, or None if not recorded.

    Returns
    -------
    ObservationParameters
        Start time, observation length, integration time and channels of
        the field.
    """
    # Measurement sets of a field share the integration time, so it is
    # taken from the first one
    integration_time = float(np.median(np.diff(timestamps[0])))

    # Timestamps mark the centre of each integration, so the observation
    # spans half an integration beyond the first and last ones
    start_time = (
        min(min(ms_timestamps) for ms_timestamps in timestamps)
        - integration_time / 2
    )
    end_time = (
        max(max(ms_timestamps) for ms_timestamps in timestamps)
        + integration_time / 2
    )

    return ObservationParameters(
        start_time=float(start_time),
        observation_length=float(end_time - start_time),
        integration_time=integration_time,
        channels=channels,
    )


def get_station_configuration(
    field_metadata: FieldMetadata,
) -> StationConfiguration:
    """
    Get the stations of a field from its run metadata.

    Every section of the report has one row per station, so each builder
    takes its stations from here.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its stations as recorded by
        load_data: MS station ID keyed by MS station name, in
        configuration order.

    Returns
    -------
    StationConfiguration
        Stations of the field.

    Raises
    ------
    RunMetadataError
        If no stations are recorded for the field.
    """

    stations = repository.get_station_configuration(field_metadata)
    if stations is not None:
        return stations

    raise RunMetadataError(
        f"No stations recorded for field {field_metadata.field_name}"
    )


def get_file_information(field_metadata: FieldMetadata) -> FileInformation:
    """
    Get the files of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its measurement sets.

    Returns
    -------
    FileInformation
        Measurement sets of the field and QA directory of the run.

    Raises
    ------
    RunMetadataError
        If no measurement sets are recorded for the field.
    """

    file_information = repository.get_file_information(field_metadata)
    if file_information is not None:
        return file_information

    raise RunMetadataError(
        f"QA metadata missing for field {field_metadata.field_name}"
    )


def get_timestamps(field_metadata: FieldMetadata) -> list[list[float]]:
    """
    Get the timestamps of the measurement sets of a field from its run
    metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with the timestamps of each of its
        measurement sets as recorded by load_data.

    Returns
    -------
    list[list[float]]
        Timestamps of each measurement set of the field, in MJD seconds,
        in any order.

    Raises
    ------
    RunMetadataError
        If no timestamps are recorded for the field.
    """

    timestamps = repository.get_timestamps(field_metadata)
    if timestamps is not None:
        return timestamps

    raise RunMetadataError(
        f"No timestamps recorded for field {field_metadata.field_name}"
    )


def get_channels(field_metadata: FieldMetadata) -> int | None:
    """
    Get the number of frequency channels of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with, optionally, its number of
        channels as recorded by load_data.

    Returns
    -------
    int or None
        Number of channels, or None if not recorded: the report shows it
        as unknown rather than failing the section.
    """

    return repository.get_channels(field_metadata)
