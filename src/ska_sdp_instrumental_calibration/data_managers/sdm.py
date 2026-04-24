"""SDM class to manage science data model"""

import os

from ska_sdp_datamodels.science_data_model.science_data_model import (
    ScienceDataModel,
)


def get_gaintable_file_path(output_dir, filename, sdm_path, purpose, field_id):
    """
    Generate the file path for a gain table.

    Parameters
    ----------
    output_dir : str or Path
        Fallback directory if no SDM path is provided.
    filename : str
        Base name of the gain table file.
    sdm_path : str or Path, optional
        Path to the Science Data Model directory.
    purpose : str
        The calibration purpose of the gain table.
    field_id : str or int
        Identifier for the observed field.

    Returns
    -------
    Path
        The resolved destination path for the gain table file.
    """

    if sdm_path is not None:
        sdm = ScienceDataModel(sdm_path)
        gaintable_path = sdm.get_calibration_table(
            field_id=field_id, purpose=purpose, file_name=filename
        )
        gaintable_path.parent.mkdir(exists_ok=True, parents=True)
        return str(gaintable_path)

    return os.path.join(output_dir, f"{field_id}_{filename}")


def prepare_qa_path(output_dir, sdm_path, **kwargs):
    """
    Initialize SDM directory structure and prepare the QA path.

    Parameters
    ----------
    output_dir : str
        Base directory used to construct the SDM path if not provided.
    sdm_path : str or None
        Path to the SDM directory. If None, it defaults to a 'sdm'
        subdirectory within output_dir.
    **kwargs : dict
        Additional keyword arguments for path preparation.

    Returns
    -------
    str
        The path to the prepared log directory.
    """
    if sdm_path is None:
        sdm_path = f"{output_dir}/sdm/"

    sdm = ScienceDataModel(sdm_path)

    if not os.path.exists(sdm_path):
        sdm.create_empty(sdm_path)

    logs_path = sdm.get_next_logs_dir("inst")
    logs_path.mkdir(parents=True, exist_ok=True)

    return logs_path
