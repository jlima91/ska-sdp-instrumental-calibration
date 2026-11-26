"""
Controller of a field's QA report sections: the builders of the report,
registered on a :class:`.section_registry.SectionRegistry`.

``section_registry``, the module's SectionRegistry, is the entry point:
``section_registry.build(SectionType, field_metadata)`` runs the builder of
that type and shows a run metadata error as an error section, so a
missing or unreadable product leaves an error in its section instead of
failing the report.

Each builder looks up the paths of the products it needs in the run
metadata and reads them with :mod:`..repository` (e.g.
:mod:`..repository.delaytable`), computes the metrics with
:mod:`..service` (e.g. :mod:`..service.cable_delays`), and returns its
section from :mod:`..view`. To add a section, put the lookup and reader
of a new product in a module of :mod:`..repository`, its metrics in a
module of :mod:`..service`, then write a ``<name>_section`` builder here
and decorate it with ``@section_registry(<SectionType>)``.

The builders are public, though only called through
``section_registry.build``: the decorator registers them, so nothing calls
them by name, and private ones would look unused. Call them through
``section_registry.build``, which shows their run metadata errors as error
sections.
"""

from ..repository.delaytable import get_delaytable_paths, read_delaytable
from ..repository.field_metadata import FieldMetadata
from ..repository.gaintable import (
    get_bandpass_gaintable_paths,
    get_final_gaintable_path,
    read_gaintable,
)
from ..repository.metadata import (
    get_channels,
    get_file_information,
    get_station_configuration,
    get_timestamps,
)
from ..repository.pre_flagging import get_pre_flagging_paths, read_pre_flagging
from ..service.bandpass_fit import bandpass_fit_quality
from ..service.cable_delays import get_cable_delays
from ..service.flagging_statistics import get_flagged_from_gaintable
from ..service.metadata_summary import get_observation_parameters
from ..service.pre_flagging import pre_flags_from_counts
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
    those recorded in the run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its stations, its final gaintable
        and the pre-flagging data of its measurement sets.

    Returns
    -------
    FlaggingStatisticsSection
        Flagging statistics of the field.

    Raises
    ------
    RunMetadataError
        If the stations are not recorded, the final gaintable is not
        recorded, was not produced or cannot be read, or the pre-flagging
        data is not recorded, was not produced or cannot be read.
    """
    gaintable = read_gaintable([get_final_gaintable_path(field_metadata)])
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

    The observation spans the timestamps of all measurement sets. The
    files, timestamps, channels and stations are read from the run
    metadata of the field.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its measurement sets, their
        timestamps, its stations and, optionally, its channels.

    Returns
    -------
    MetadataSection
        File information, observation parameters and stations of the
        field.

    Raises
    ------
    RunMetadataError
        If the measurement sets, timestamps or stations are not recorded.
    """

    return MetadataSection(
        file_information=get_file_information(field_metadata),
        observation_parameters=get_observation_parameters(
            get_timestamps(field_metadata), get_channels(field_metadata)
        ),
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
    time. The stations are those recorded in the run metadata; a station
    missing from the delay tables gets a row with only its ID.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its stations and the delay table
        of each of its measurement sets.

    Returns
    -------
    CableDelaysSection
        Cable delays of the field.

    Raises
    ------
    RunMetadataError
        If no delay table is recorded, one was not produced or cannot be
        read, the delay tables do not have the same stations, or the
        stations are not recorded.
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
    sets, concatenated along time. The stations are those recorded in the
    run metadata; a station missing from the gaintables gets a row with
    only its ID. See :func:`..service.bandpass_fit.bandpass_fit_quality`.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its stations and the bandpass
        gaintable of each of its measurement sets.
    phase_fit_threshold : float
        Phase fit quality below which a station fails QA.
    gain_fit_threshold : float
        Gain amplitude fit quality below which a station fails QA.

    Returns
    -------
    BandpassFitSection
        Bandpass fit metrics of the field.

    Raises
    ------
    RunMetadataError
        If no bandpass gaintable is recorded, one was not produced or
        cannot be read, the gaintables do not have the same stations and
        channels, or the stations are not recorded.
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
