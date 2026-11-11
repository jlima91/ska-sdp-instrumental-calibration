"""SDM class to manage science data model"""

import os
from pathlib import Path

from ska_sdp_datamodels.science_data_model.science_data_model import (
    ScienceDataModel,
)

GAINTABLE_FILE_NAME = "gaintable.h5parm"
GAINTABLE_PURPOSE = "bandpass"


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


def get_qa_metric_dir(output_path: str, sdm_path: str | None = None) -> str:
    """
    Get the directory to write the QA metric of the latest INST run to.

    Parameters
    ----------
    output_path
        Output directory of the INST run. Used if no SDM path is provided.
    sdm_path
        Path to an existing SDM directory. Takes precedence over
        ``output_path``.

    Returns
    -------
        The logs directory of the latest INST run if an SDM path is
        provided, else ``output_path``.
    """
    if sdm_path is None:
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Provided output path {output_path} does not exist. "
                "Please provide a valid path."
            )
        return str(output_path)

    if not os.path.exists(sdm_path):
        raise FileNotFoundError(
            f"Provided SDM path {sdm_path} does not exist. "
            "Please provide a valid path."
        )

    latest_run_path = ScienceDataModel(sdm_path).get_latest_logs_dir("inst")
    if latest_run_path is None:
        raise FileNotFoundError(
            f"No latest INST logs found in SDM {sdm_path}."
        )

    return str(latest_run_path)


def get_final_gaintable_path(
    qa_metric_dir: str, field_id: str, sdm_path: str | None = None
) -> str:
    """
    Get the path of the final gaintable of a field
    Parameters
    ----------
    qa_metric_dir
        Directory of the INST run. Used if no SDM path is provided.
    field_id
        Field the gaintable belongs to.
    sdm_path
        Path to the SDM directory, if the INST run is in an SDM.

    Returns
    -------
        The SDM calibration table path if an SDM path is provided, else
        <qa_metric_dir>/<field_id>_gaintable.h5parm.
    """
    if sdm_path is not None:
        return str(
            ScienceDataModel(sdm_path).get_calibration_table(
                field_id=field_id,
                purpose=GAINTABLE_PURPOSE,
                file_name=GAINTABLE_FILE_NAME,
            )
        )

    return os.path.join(qa_metric_dir, f"{field_id}_{GAINTABLE_FILE_NAME}")
