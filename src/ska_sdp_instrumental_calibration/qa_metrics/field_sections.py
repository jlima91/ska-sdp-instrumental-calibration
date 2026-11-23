"""
Sections of a field's QA report, built from its run metadata.

``section_builder``, the module's SectionBuilder, is the entry point:
``section_builder.build(SectionType, field_metadata)`` runs the builder of
that type and shows a run metadata error as an error section, so a
missing or unreadable product leaves an error in its section instead of
failing the report.

Each builder looks up the paths of the products it needs in the run
metadata, reads them through :mod:`.products`, computes the metrics with
:mod:`.domain` and returns its section. To add a section, write a
``<name>_section`` builder and decorate it with
``@section_builder(<SectionType>)``.

The builders are public, though only called through
``section_builder.build``: the decorator registers them, so nothing calls
them by name, and private ones would look unused. Call them through
``section_builder.build``, which shows their run metadata errors as error
sections.
"""

import numpy as np

from .domain.qa_metadata import (
    FileInformation,
    ObservationParameters,
    PreFlagging,
    StationConfiguration,
)
from .domain.qa_metric import (
    BandpassFitMetrics,
    CableDelays,
    FlaggingStatistics,
)
from .exceptions import RunMetadataError
from .products import read_delaytable, read_gaintable, read_pre_flagging
from .qa_section import QASection
from .run_metadata import FieldMetadata
from .section_builder import SectionBuilder
from .sections import (
    CableDelaysSection,
    FlaggingStatisticsSection,
    MetadataSection,
)
from .tables import (
    CABLE_DELAYS_COLUMNS,
    FLAGGING_STATISTICS_COLUMNS,
    build_station_rows,
)

section_builder = SectionBuilder()


@section_builder(FlaggingStatisticsSection)
def flagging_statistics_section(
    field_metadata: FieldMetadata,
) -> FlaggingStatisticsSection:
    """
    Build the flagging statistics section of a field.

    One row per station with the percentage of its gain solutions that
    cannot be solved before calibration, from the pre-flagging data of its
    measurement sets, and that are flagged after calibration, from its
    final gaintable, and the totals over all stations. The stations are
    read off the final gaintable.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its final gaintable and the
        pre-flagging data of its measurement sets.

    Returns
    -------
    FlaggingStatisticsSection
        Flagging statistics of the field.

    Raises
    ------
    RunMetadataError
        If the final gaintable is not recorded, was not produced or cannot
        be read, or the pre-flagging data is not recorded, was not
        produced or cannot be read.
    """
    gaintable = read_gaintable([_get_final_gaintable(field_metadata)])
    station_names = [str(name) for name in gaintable.antenna_name.values]

    pre_flagging = PreFlagging.from_counts(
        read_pre_flagging(
            _get_pre_flagging_paths(field_metadata), len(station_names)
        ),
        # Pre-flagging counts are in station order, as the gaintable
        station_names,
    )
    flagging_statistics = FlaggingStatistics.from_gaintable(
        gaintable, flagged_before=pre_flagging.unsolvable_gains
    )

    return FlaggingStatisticsSection(
        columns=FLAGGING_STATISTICS_COLUMNS,
        rows=build_station_rows(
            StationConfiguration.from_station_names(station_names),
            flagging_statistics.values_per_station,
        ),
        totals=flagging_statistics.totals,
    )


@section_builder(MetadataSection)
def metadata_section(field_metadata: FieldMetadata) -> MetadataSection:
    """
    Build the metadata summary section of a field.

    The timestamps of the measurement sets are joined along time. The
    channels are read from the run metadata and the stations off the final
    gaintable of the field.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its measurement sets, their
        timestamps, its final gaintable and, optionally, its channels.

    Returns
    -------
    MetadataSection
        File information, observation parameters and stations of the
        field.

    Raises
    ------
    RunMetadataError
        If the measurement sets, timestamps or final gaintable are not
        recorded, or the final gaintable was not produced or cannot be
        read.
    """
    input_ms = field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS)
    timestamps = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.TIMESTAMPS
    )
    gaintable_path = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.FINAL_GAINTABLE
    )
    channels = field_metadata.get_data_for(QASection.METADATA_SUMMARY.CHANNELS)

    if not input_ms or not timestamps or not gaintable_path:
        raise RunMetadataError(
            f"QA metadata missing for field {field_metadata.field_name}"
        )

    gaintable = read_gaintable([gaintable_path])

    return MetadataSection(
        file_information=FileInformation(
            input_ms=list(input_ms), output_path=field_metadata.qa_dir
        ),
        observation_parameters=_get_observation_parameters(
            timestamps, channels=channels
        ),
        station_configuration=StationConfiguration.from_station_names(
            gaintable.antenna_name.values
        ),
    )


@section_builder(CableDelaysSection)
def cable_delays_section(
    field_metadata: FieldMetadata,
) -> CableDelaysSection:
    """
    Build the cable delays section of a field.

    One row per station with its median X and Y delay, in nanoseconds,
    over the delay tables of its measurement sets, concatenated along
    time. The stations are read off the delay tables.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with the delay table of each of its
        measurement sets.

    Returns
    -------
    CableDelaysSection
        Cable delays of the field.

    Raises
    ------
    RunMetadataError
        If no delay table is recorded, one was not produced or cannot be
        read, or the delay tables do not have the same stations.
    """
    delaytable = read_delaytable(_get_delaytable_paths(field_metadata))
    cable_delays = CableDelays.from_delaytable(delaytable)

    return CableDelaysSection(
        columns=CABLE_DELAYS_COLUMNS,
        rows=build_station_rows(
            StationConfiguration.from_station_names(
                delaytable.antenna_name.values
            ),
            cable_delays.values_per_station,
        ),
    )


def _get_final_gaintable(field_metadata: FieldMetadata) -> str:
    """
    Get the final gaintable path of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field.

    Returns
    -------
    str
        Path of the final gaintable of the field.

    Raises
    ------
    RunMetadataError
        If no final gaintable is recorded for the field.
    """
    gaintable = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.FINAL_GAINTABLE
    )
    if not gaintable:
        raise RunMetadataError(
            f"No final gaintable recorded for field "
            f"{field_metadata.field_name}"
        )
    return gaintable


def _get_pre_flagging_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the pre-flagging data files of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one pre-flagging data file
        recorded per MS prefix.

    Returns
    -------
    list[str]
        Pre-flagging data file of each measurement set of the field. Empty
        if none is recorded: the reader raises for it.
    """
    pre_flagging = (
        field_metadata.get_data_for(QASection.FLAGGING.PRE_FLAGGING) or {}
    )
    return list(pre_flagging.values())


def _get_delaytable_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the delay tables of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one delay table recorded per MS
        prefix.

    Returns
    -------
    list[str]
        Delay table of each measurement set of the field. Empty if none is
        recorded: the reader raises for it.
    """
    delaytables = field_metadata.get_data_for(QASection.DELAY.DELAYTABLE) or {}
    return list(delaytables.values())


def _get_observation_parameters(
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


def get_bandpass_fit_metrics(
    field_metadata: FieldMetadata,
) -> BandpassFitMetrics:
    """
    Get the bandpass fit metrics of a field. Not implemented yet.

    To be replaced by a ``bandpass_fit_metrics_section`` builder,
    decorated with ``@section_builder(BandpassFitMetricsSection)``.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field.

    Returns
    -------
    BandpassFitMetrics
        Bandpass fit metrics of the field.

    Raises
    ------
    NotImplementedError
        Always, until the metrics are implemented.
    """
    raise NotImplementedError("Not Implemented")
