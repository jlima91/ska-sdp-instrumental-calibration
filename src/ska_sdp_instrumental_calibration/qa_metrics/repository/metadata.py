"""
Metadata of a field recorded by load_data: its stations, its measurement
sets, their timestamps and its number of channels, looked up in its run
metadata.
"""

from ..domain.metadata import FileInformation, StationConfiguration
from .field_metadata import FieldMetadata
from .manifest_schema import ManifestSchema


def get_station_configuration(
    field_metadata: FieldMetadata,
) -> StationConfiguration | None:
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
        return None
    return StationConfiguration(stations)


def get_file_information(
    field_metadata: FieldMetadata,
) -> FileInformation | None:
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
    input_ms = field_metadata.get_data_for(ManifestSchema.METADATA_SUMMARY.MS)

    if not input_ms:
        return None
    return FileInformation(
        input_ms=list(input_ms), output_path=field_metadata.qa_dir
    )


def get_timestamps(
    field_metadata: FieldMetadata,
) -> list[list[float]] | None:
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
    timestamps = field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.TIMESTAMPS
    )
    if not timestamps:
        return None
    return timestamps


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
    return field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.CHANNELS
    )
