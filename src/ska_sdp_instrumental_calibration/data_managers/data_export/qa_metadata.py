import glob
import logging
import os

from ...exceptions import QAMetadataError
from ...xarray_processors.vis_flagging import flagged_time_count
from ..schema.qa_metadata import QAMetadata

logger = logging.getLogger(__name__)

QA_METADATA_FILE_NAME = "qa-metadata-data.json"


def export_visibility_metadata(metadata_path, upstream_outputs):
    raise NotImplementedError("Export QA metadata not implemented")


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


def read_qa_metadata_from_dir(qa_dir: str) -> list[QAMetadata]:
    """
    Read the QA metadata of an INST run, one per field.

    Parameters
    ----------
    qa_dir: str
        QA directory of the run: the SDM logs directory or the output
        directory.

    Returns
    -------
    list[QAMetadata]
        QA metadata of each <field>_qa-metadata-data.json file in qa_dir.

    Raises
    ------
    QAMetadataError
        If no QA metadata file is found, or a file is malformed or
        incomplete.
    """
    qa_metadata_paths = sorted(
        glob.glob(os.path.join(qa_dir, f"*_{QA_METADATA_FILE_NAME}"))
    )
    if not qa_metadata_paths:
        raise QAMetadataError(f"No QA metadata found in {qa_dir}")

    return [QAMetadata.read_from_json(path) for path in qa_metadata_paths]
