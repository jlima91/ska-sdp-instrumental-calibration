"""
Metadata summary of a field: its stations, looked up in its run metadata,
and its observation parameters, from the timestamps of its measurement
sets.
"""

import numpy as np

from ..domain.metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from ..exceptions import RunMetadataError
from ..repository.field_metadata import FieldMetadata
from ..repository.manifest_schema import ManifestSchema


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
    stations = field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.STATION_CONFIGURATION
    )
    if not stations:
        raise RunMetadataError(
            f"No stations recorded for field {field_metadata.field_name}"
        )
    return StationConfiguration(stations)


def get_file_information(field_metadata) -> FileInformation:
    input_ms = field_metadata.get_data_for(ManifestSchema.METADATA_SUMMARY.MS)

    if not input_ms:
        raise RunMetadataError(
            f"QA metadata missing for field {field_metadata.field_name}"
        )
    return FileInformation(
        input_ms=list(input_ms), output_path=field_metadata.qa_dir
    )


def get_observation_parameters(field_metadata) -> ObservationParameters:
    """
    Get the observation parameters of a field from the timestamps of its
    measurement sets.

    The integration time is the median spacing of the timestamps of the
    first measurement set. The observation spans the timestamps of all
    measurement sets, which mark the centre of each integration.

    Parameters
    ----------
    timestamps : list[list[float]]
        Timestamps of each measurement set of the field, in MJD seconds,
        in any order.
    channels : int or None
        Number of frequency channels, or None if not recorded.

    Returns
    -------
    ObservationParameters
        Start time, observation length, integration time and channels of
        the field.
    """
    timestamps = field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.TIMESTAMPS
    )
    channels = field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.CHANNELS
    )

    if not timestamps:
        raise RunMetadataError(
            f"QA metadata missing for field {field_metadata.field_name}"
        )

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
