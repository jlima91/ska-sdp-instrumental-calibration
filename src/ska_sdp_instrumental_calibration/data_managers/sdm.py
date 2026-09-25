"""SDM class to manage science data model"""

import os
from pathlib import Path

from ska_sdp_datamodels.science_data_model.science_data_model import (
    ScienceDataModel,
)


def get_gaintable_file_path(
    output_dir: str | Path,
    filename: str,
    sdm_path: str | Path | None,
    purpose: str,
    field_id: str,
) -> str:
    """
    Generate the file path for a gain table.

    Parameters
    ----------
    output_dir
        Fallback directory if no SDM path is provided.
    filename
        Base name of the gain table file.
    sdm_path
        Path to the Science Data Model directory.
        If None, gaintable will be written to the output_dir.
    purpose
        The calibration purpose of the gain table.
    field_id
        Identifier for the observed field.

    Returns
    -------
        The resolved destination path for the gain table file.
    """

    if sdm_path is not None:
        sdm = ScienceDataModel(sdm_path)
        gaintable_path = sdm.get_calibration_table(
            field_id=field_id, purpose=purpose, file_name=filename
        )
        gaintable_path.parent.mkdir(exist_ok=True, parents=True)
        return str(gaintable_path)

    return os.path.join(output_dir, f"{field_id}_{filename}")


def prepare_qa_path(
    output_dir: str, sdm_path: str | None = None, **kwargs
) -> str:
    """
    Initialize SDM directory structure and prepare the QA path.

    Parameters
    ----------
    output_dir
        Default output directory passed to the function by piper.
    sdm_path
        Path to the SDM directory as provided
        from the CLI option ``--sdm-path``.
        If None, then the output_dir will be returned as QA path.
        If a string, then it must be an existing SDM directory
        with valid structure.
    **kwargs
        Additional CLI arguments passed by piper.

    Returns
    -------
        The QA path where pipeline will dump its QA outputs
    """
    if sdm_path is None:
        return str(output_dir)

    if not os.path.exists(sdm_path):
        raise FileNotFoundError(
            f"Provided SDM path {sdm_path} does not exist. "
            "Please provide a valid path."
        )

    sdm = ScienceDataModel(sdm_path)
    logs_path = sdm.get_next_logs_dir("inst")
    logs_path.mkdir(parents=True, exist_ok=True)

    return str(logs_path)
