"""
Controller of a field's QA report sections: the builders of the report,
registered on a :class:`.section_registry.SectionRegistry`.

``section_registry``, the module's SectionRegistry, is the entry point:
``section_registry.build(SectionType, field_metadata)`` runs the builder of
that type and shows a run metadata error as an error section, so a
missing or unreadable product leaves an error in its section instead of
failing the report.

Each builder looks up the paths of the products it needs in the run
metadata, reads them and computes the metrics with its module of
:mod:`..service` (e.g. :mod:`..service.cable_delays`), and returns its
section from :mod:`..view`. To add a section, put its lookups, readers
and metrics in a module of :mod:`..service`, then write a
``<name>_section`` builder here and decorate it with
``@section_registry(<SectionType>)``.

The builders are public, though only called through
``section_registry.build``: the decorator registers them, so nothing calls
them by name, and private ones would look unused. Call them through
``section_registry.build``, which shows their run metadata errors as error
sections.
"""

from ..repository import FieldMetadata
from ..service.bandpass_fit import (
    bandpass_fit_quality,
    get_bandpass_gaintable_paths,
)
from ..service.cable_delays import (
    get_cable_delays,
    get_delaytable_paths,
    read_delaytable,
)
from ..service.flagging_statistics import (
    get_final_gaintable,
    get_flagged_from_gaintable,
)
from ..service.gaintable import read_gaintable
from ..service.metadata_summary import (
    get_file_information,
    get_observation_parameters,
    get_station_configuration,
)
from ..service.pre_flagging import (
    get_pre_flagging_paths,
    pre_flags_from_counts,
    read_pre_flagging,
)
from ..view.sections import (
    BandpassFitSection,
    CableDelaysSection,
    FlaggingStatisticsSection,
    MetadataSection,
)
from ..view.tables import (
    CABLE_DELAYS_COLUMNS,
    FLAGGING_STATISTICS_COLUMNS,
    build_station_rows,
    get_bandpass_fit_metrics_columns,
)
from .section_registry import SectionRegistry

section_registry = SectionRegistry()


@section_registry(FlaggingStatisticsSection)
def flagging_statistics_section(
    field_metadata: FieldMetadata, **kwargs
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

    pre_flagging = pre_flags_from_counts(
        read_pre_flagging(
            get_pre_flagging_paths(field_metadata), len(station_names)
        ),
        # Pre-flagging counts are in configuration order, as the stations
        station_names,
    )
    flagging_statistics = get_flagged_from_gaintable(
        gaintable, flagged_before=pre_flagging.unsolvable_gains
    )

    return FlaggingStatisticsSection(
        columns=FLAGGING_STATISTICS_COLUMNS,
        rows=build_station_rows(
            station_configuration,
            flagging_statistics,
        ),
        totals=flagging_statistics.totals,
    )


@section_registry(MetadataSection)
def metadata_section(
    field_metadata: FieldMetadata, **kwargs
) -> MetadataSection:
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

    return MetadataSection(
        file_information=get_file_information(field_metadata),
        observation_parameters=get_observation_parameters(field_metadata),
        station_configuration=get_station_configuration(field_metadata),
    )


@section_registry(CableDelaysSection)
def cable_delays_section(
    field_metadata: FieldMetadata, **kwargs
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
    cable_delays = get_cable_delays(delaytable)

    return CableDelaysSection(
        columns=CABLE_DELAYS_COLUMNS,
        rows=build_station_rows(
            get_station_configuration(field_metadata),
            cable_delays,
        ),
    )


@section_registry(BandpassFitSection)
def bandpass_fit_section(
    field_metadata: FieldMetadata,
    *,
    phase_fit_threshold,
    gain_fit_threshold,
    **kwargs,
) -> BandpassFitSection:
    """
    Build the bandpass fit section of a field.

    One row per station with the phase and gain amplitude fit quality in
    X and Y polarisation, over the bandpass gaintables of its measurement
    sets, concatenated along time. See
    :func:`..service.bandpass_fit.bandpass_fit_quality`.

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
    bandpass_fit = bandpass_fit_quality(gaintable)

    return BandpassFitSection(
        columns=get_bandpass_fit_metrics_columns(
            phase_fit_threshold, gain_fit_threshold
        ),
        rows=build_station_rows(
            get_station_configuration(field_metadata),
            bandpass_fit,
        ),
    )
