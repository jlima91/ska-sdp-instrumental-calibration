"""
Queries that build the domain objects of a field's QA report from its run
metadata.
"""

import numpy as np

from ..data_managers.h5parm import read_h5parm_gains
from ..exceptions import QAMetadataError
from .domain import QAMetadata
from .domain.qa_metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from .domain.qa_metric import (
    BandpassFitMetrics,
    CableDelays,
    FlaggingStatistics,
)
from .qa_section import QASection
from .run_metadata import FieldMetadata


def get_cable_delays(
    field_metadata: FieldMetadata,
) -> CableDelays:
    raise NotImplementedError("Not Implemented")


def get_bandpass_fit_metrics(
    field_metadata: FieldMetadata,
) -> BandpassFitMetrics:
    raise NotImplementedError("Not Implemented")


def get_flagging_statistics(
    field_metadata: FieldMetadata,
) -> FlaggingStatistics:
    raise NotImplementedError("Not Implemented")


def get_qa_metadata(field_metadata: FieldMetadata) -> QAMetadata:
    """
    Get the QA metadata of a field from its run metadata.

    The timestamps of the measurement sets are joined along time. The
    channels and stations are read off the gaintable of the field.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with its measurement sets, their
        timestamps and the gaintable of the field.

    Returns
    -------
    QAMetadata
        QA metadata of the field.

    Raises
    ------
    QAMetadataError
        If the measurement sets, timestamps or gaintable are not recorded,
        the gaintable was not produced, or it has no frequency or antenna
        axis.
    """
    input_ms = field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS)
    timestamps = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.TIMESTAMPS
    )
    gaintables = field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.FINAL_GAINTABLE
    )

    channels = field_metadata.get_data_for(QASection.METADATA_SUMMARY.CHANNELS)
    if channels is not None:
        channels = channels[0]

    if not input_ms or not timestamps or not gaintables:
        raise QAMetadataError(
            f"QA metadata missing for field {field_metadata.field_name}"
        )

    station_names = _read_gaintable_axes(gaintables[0])

    return QAMetadata(
        source_name=field_metadata.field_name,
        file_information=FileInformation(
            input_ms=list(input_ms), output_path=field_metadata.qa_dir
        ),
        observation_parameters=_get_observation_parameters(
            timestamps, channels=channels
        ),
        station_configuration=StationConfiguration(
            {
                str(station_id): str(name)
                for station_id, name in enumerate(station_names)
            }
        ),
    )


def _read_gaintable_axes(
    gaintable_path: str,
) -> list[str]:
    """
    Read the frequency and antenna axes of a gaintable.
    """
    try:
        *_, soltab = read_h5parm_gains(gaintable_path)
    except FileNotFoundError as error:
        # The run metadata lists the products a run should write, so a
        # listed gaintable can be missing if its export failed
        raise QAMetadataError(
            f"Final gaintable {gaintable_path} was not produced"
        ) from error

    if soltab.frequency is None or soltab.antenna is None:
        raise QAMetadataError(
            f"Gaintable {gaintable_path} has no frequency or antenna axis"
        )

    return soltab.antenna


def _get_observation_parameters(
    timestamps: list[list[float]], channels: int
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
