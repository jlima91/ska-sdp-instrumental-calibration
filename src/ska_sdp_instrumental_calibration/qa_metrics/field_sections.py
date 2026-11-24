"""
Sections of a field's QA report, built from its run metadata.

``section_builder``, the module's SectionBuilder, is the entry point:
``section_builder.build(SectionType, field_metadata)`` runs the builder of
that type and shows a run metadata error as an error section, so a
missing or unreadable product leaves an error in its section instead of
failing the report.

Each builder looks up the paths of the products it needs in the run
metadata with its ``<name>_metric`` module of :mod:`.service` (e.g.
:mod:`.service.bandpass_fit_metric`), reads them through
:mod:`.products`, computes the metrics with its service or
:mod:`.domain`, and returns its section. To add a section, put its
lookups and metrics in a ``<name>_metric`` module of :mod:`.service`,
export them from the package, then write a ``<name>_section`` builder
here and decorate it with ``@section_builder(<SectionType>)``.

The builders are public, though only called through
``section_builder.build``: the decorator registers them, so nothing calls
them by name, and private ones would look unused. Call them through
``section_builder.build``, which shows their run metadata errors as error
sections.
"""

from .domain.qa_metadata import FileInformation, PreFlagging
from .domain.qa_metric import CableDelays, FlaggingStatistics
from .exceptions import RunMetadataError
from .products import read_delaytable, read_gaintable, read_pre_flagging
from .qa_section import QASection
from .run_metadata import FieldMetadata
from .section_builder import SectionBuilder
from .sections import (
    BandpassFitSection,
    CableDelaysSection,
    FlaggingStatisticsSection,
    MetadataSection,
)
from .service import (
    bandpass_fit_quality,
    get_bandpass_gaintable_paths,
    get_delaytable_paths,
    get_final_gaintable,
    get_observation_parameters,
    get_pre_flagging_paths,
    get_station_configuration,
)
from .tables import (
    BANDPASS_FIT_METRICS_COLUMNS,
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
    gaintable = read_gaintable([get_final_gaintable(field_metadata)])
    station_configuration = get_station_configuration(field_metadata)
    station_names = list(station_configuration.station_names)

    pre_flagging = PreFlagging.from_counts(
        read_pre_flagging(
            get_pre_flagging_paths(field_metadata), len(station_names)
        ),
        # Pre-flagging counts are in configuration order, as the stations
        station_names,
    )
    flagging_statistics = FlaggingStatistics.from_gaintable(
        gaintable, flagged_before=pre_flagging.unsolvable_gains
    )

    return FlaggingStatisticsSection(
        columns=FLAGGING_STATISTICS_COLUMNS,
        rows=build_station_rows(
            station_configuration,
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
    channels = field_metadata.get_data_for(QASection.METADATA_SUMMARY.CHANNELS)

    if not input_ms or not timestamps:
        raise RunMetadataError(
            f"QA metadata missing for field {field_metadata.field_name}"
        )

    return MetadataSection(
        file_information=FileInformation(
            input_ms=list(input_ms), output_path=field_metadata.qa_dir
        ),
        observation_parameters=get_observation_parameters(
            timestamps, channels=channels
        ),
        station_configuration=get_station_configuration(field_metadata),
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
    delaytable = read_delaytable(get_delaytable_paths(field_metadata))
    cable_delays = CableDelays.from_delaytable(delaytable)

    return CableDelaysSection(
        columns=CABLE_DELAYS_COLUMNS,
        rows=build_station_rows(
            get_station_configuration(field_metadata),
            cable_delays.values_per_station,
        ),
    )


@section_builder(BandpassFitSection)
def bandpass_fit_section(
    field_metadata: FieldMetadata,
) -> BandpassFitSection:
    """
    Build the bandpass fit section of a field.

    One row per station with the phase and gain amplitude fit quality in
    X and Y polarisation, over the bandpass gaintables of its measurement
    sets, concatenated along time. See
    :func:`.bandpass_fit_metric.bandpass_fit_quality`.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with the bandpass gaintable of each of
        its measurement sets.

    Returns
    -------
    BandpassFitSection
        Bandpass fit metrics of the field.

    Raises
    ------
    RunMetadataError
        If no bandpass gaintable is recorded, one was not produced or
        cannot be read, or the gaintables do not have the same stations
        and channels.
    """
    gaintable = read_gaintable(get_bandpass_gaintable_paths(field_metadata))

    return BandpassFitSection(
        columns=BANDPASS_FIT_METRICS_COLUMNS,
        rows=build_station_rows(
            get_station_configuration(field_metadata),
            bandpass_fit_quality(gaintable),
        ),
    )
