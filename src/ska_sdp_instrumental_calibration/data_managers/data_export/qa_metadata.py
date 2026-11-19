import logging
import os

import numpy as np

from ...qa_metrics.domain import (
    QA_METADATA_FILE_NAME,
    ObservationParameters,
    QAMetadata,
)
from ...xarray_processors.vis_flagging import flagged_time_count

logger = logging.getLogger(__name__)


def export_visibility_metadata(metadata_path, upstream_outputs):
    raise NotImplementedError("Export QA metadata not implemented")


def get_observation_timestamp(vis):
    """
    Get the observation parameters of a field from the QA data of its
    measurement sets, joined along time.

    Parameters
    ----------
    ms_qa_data : list[dict]
        QA data of each measurement set of the field, from
        get_ms_qa_data.

    Returns
    -------
    ObservationParameters
        Observation parameters of the field.
    """
    # Measurement sets of a field share channels and integration time,
    # so they are taken from the first one
    first_ms = ms_qa_data[0]
    integration_time = np.median(np.diff(first_ms["timestamps"]))

    start_time = (
        min(ms["timestamps"].min() for ms in ms_qa_data) - integration_time / 2
    )
    end_time = (
        max(ms["timestamps"].max() for ms in ms_qa_data) + integration_time / 2
    )

    return ObservationParameters(
        start_time=float(start_time),
        observation_length=float(end_time - start_time),
        integration_time=float(integration_time),
        channels=int(first_ms["channels"]),
    )


def get_ms_qa_data(vis, input_ms: str) -> dict:
    """
    Get the visibility data of one measurement set needed for the QA
    metadata.

    Parameters
    ----------
    vis: Visibility
        Visibility dataset of the measurement set.
    input_ms: str
        Path of the measurement set.

    Returns
    -------
    dict
        QA data of the measurement set.
    """

    return {
        "input_ms": input_ms,
        "timestamps": vis.time.data,
        "channels": vis.sizes["frequency"],
        "station_names": vis.attrs["configuration"].names.data,
        "antenna1": vis.antenna1.data,
        "antenna2": vis.antenna2.data,
        "flagged_time_count": flagged_time_count(vis).data,
        "ntimes": vis.sizes["time"],
    }


def write_qa_metadata_to_dir(
    field_id: str, ms_qa_data: list[dict], qa_dir: str
):
    """
    Write the QA metadata of the measurement sets of one field to
    <qa_dir>/<field_id>_qa-metadata-data.json.

    Parameters
    ----------
    field_id: str
        Field name. Saved as the source name.
    ms_qa_data: list[dict]
        QA data of each measurement set of the field, from get_ms_qa_data.
    qa_dir: str
        QA directory of the run: the SDM logs directory or the output
        directory. Saved as the output path.
    """

    qa_metadata_path = os.path.join(
        qa_dir, f"{field_id}_{QA_METADATA_FILE_NAME}"
    )
    QAMetadata.from_ms_qa_data(field_id, ms_qa_data, qa_dir).write_to_json(
        qa_metadata_path
    )

    logger.info("QA metadata written to %s", qa_metadata_path)
