"""
Sections of a field's QA report, built from its run metadata.

:func:`build` is the entry point: it runs the builder of a section
type and shows a run metadata error as an error section, so a missing
product leaves an error in its section instead of failing the report.
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
from .exceptions import QAReportException, RunMetadataError
from .products import read_gaintable, read_pre_flagging
from .qa_section import QASection
from .run_metadata import FieldMetadata
from .sections import (
    ErrorSection,
    FlaggingStatisticsSection,
    MetadataSection,
    Section,
)
from .tables import FLAGGING_STATISTICS_COLUMNS, build_station_rows


def get_cable_delays(
    field_metadata: FieldMetadata,
) -> CableDelays:
    raise NotImplementedError("Not Implemented")


def get_bandpass_fit_metrics(
    field_metadata: FieldMetadata,
) -> BandpassFitMetrics:
    raise NotImplementedError("Not Implemented")


def build(
    section_type: type[Section], field_metadata: FieldMetadata
) -> Section:
    """
    Build a section of the QA report of a field from its run metadata.

    Only run metadata errors, i.e. missing or unreadable products, are
    shown as an error section. Any other error is a bug, so it is raised
    rather than shown in the report.

    Parameters
    ----------
    section_type : type[Section]
        Type of the section to build, e.g. ``MetadataSection``.
    field_metadata : FieldMetadata
        Run metadata of the field.

    Returns
    -------
    Section
        Section of the type given, or an ErrorSection with its title if a
        product it needs is not recorded or cannot be read.

    Raises
    ------
    KeyError
        If no builder is registered for the section type.
    """
    builder = _SECTION_BUILDERS[section_type]
    try:
        return builder(field_metadata)
    except RunMetadataError as error:
        return ErrorSection(
            section_type.title,
            QAReportException(
                f"{section_type.title} of field "
                f"{field_metadata.field_name} cannot be built",
                cause=error,
            ),
        )


def _flagging_statistics_section(
    field_metadata: FieldMetadata,
) -> FlaggingStatisticsSection:
    """
    Flagging statistics section of a field: one row per station with the
    percentage of its gain solutions that cannot be solved before
    calibration, from the pre-flagging data of its measurement sets, and
    flagged after calibration, from its final gaintable, and the totals
    over all stations.

    Raises
    ------
    RunMetadataError
        If the final gaintable is not recorded, was not produced or has no
        antenna axis, or the pre-flagging data cannot be read.
    """
    _, weight, _, soltab = read_gaintable(_get_final_gaintable(field_metadata))
    station_names = [str(name) for name in soltab.antenna]

    pre_flagging = PreFlagging.from_counts(
        read_pre_flagging(
            _get_pre_flagging_paths(field_metadata), len(station_names)
        ),
        # Pre-flagging counts are in station order, as the gaintable
        station_names,
    )
    flagging_statistics = FlaggingStatistics.from_weights(
        weight, station_names, flagged_before=pre_flagging.unsolvable_gains
    )

    return FlaggingStatisticsSection(
        columns=FLAGGING_STATISTICS_COLUMNS,
        rows=build_station_rows(
            StationConfiguration.from_station_names(station_names),
            flagging_statistics.values_per_station(),
        ),
        totals=flagging_statistics.totals(),
    )


def _get_final_gaintable(field_metadata: FieldMetadata) -> str:
    """
    Final gaintable path of a field, raising RunMetadataError if it is not
    recorded.
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
    Pre-flagging data files of all measurement sets of a field, recorded
    as one path per MS prefix.
    """
    pre_flagging = (
        field_metadata.get_data_for(QASection.FLAGGING.PRE_FLAGGING) or {}
    )
    return list(pre_flagging.values())


def _metadata_section(field_metadata: FieldMetadata) -> MetadataSection:
    """
    Metadata summary section of a field.

    The timestamps of the measurement sets are joined along time. The
    channels are read from the run metadata and the stations off the final
    gaintable of the field.

    Raises
    ------
    RunMetadataError
        If the measurement sets, timestamps or gaintable are not recorded,
        the gaintable was not produced, or it has no antenna axis.
    """
    input_ms = field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS)
    timestamps = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.TIMESTAMPS
    )
    gaintable = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.FINAL_GAINTABLE
    )
    channels = field_metadata.get_data_for(QASection.METADATA_SUMMARY.CHANNELS)

    if not input_ms or not timestamps or not gaintable:
        raise RunMetadataError(
            f"QA metadata missing for field {field_metadata.field_name}"
        )

    *_, soltab = read_gaintable(gaintable)

    return MetadataSection(
        file_information=FileInformation(
            input_ms=list(input_ms), output_path=field_metadata.qa_dir
        ),
        observation_parameters=_get_observation_parameters(
            timestamps, channels=channels
        ),
        station_configuration=StationConfiguration.from_station_names(
            soltab.antenna
        ),
    )


def _get_observation_parameters(
    timestamps: list[list[float]], channels: int | None
) -> ObservationParameters:
    """
    Observation parameters of a field, from the timestamps of each of its
    measurement sets.
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


# Builder of each section type, run by build
_SECTION_BUILDERS = {
    MetadataSection: _metadata_section,
    FlaggingStatisticsSection: _flagging_statistics_section,
}
