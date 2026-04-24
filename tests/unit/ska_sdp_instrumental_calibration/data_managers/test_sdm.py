"""Test for sdm"""

import tempfile
from pathlib import Path

from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.sdm import (
    get_gaintable_file_path,
    prepare_qa_path,
)


def test_should_prepare_qa_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        assert prepare_qa_path(temp_dir, sdm_path=None) == Path(
            f"{temp_dir}/sdm/logs/01-inst"
        )

        sdm_root = Path(f"{temp_dir}/sdm_root")
        inst_log_path = Path(f"{sdm_root}/logs/01-inst")
        assert prepare_qa_path(None, sdm_path=sdm_root) == inst_log_path


@patch(
    "ska_sdp_instrumental_calibration.data_managers.sdm.os.path.exists",
    return_value=True,
)
@patch("ska_sdp_instrumental_calibration.data_managers.sdm.ScienceDataModel")
def test_should_prepare_and_create_qa_path(sdm_mock, exists_mock):
    sdm_mock.return_value = sdm_mock
    prepare_qa_path(None, sdm_path="sdm_root")

    sdm_mock.create_empty.assert_not_called()


def test_should_get_gaintable_file_path_without_sdm():

    assert (
        get_gaintable_file_path(
            "output_dir", "filename", None, "purpose", "field_id"
        )
        == "output_dir/field_id_filename"
    )


@patch("ska_sdp_instrumental_calibration.data_managers.sdm.ScienceDataModel")
def test_should_get_gaintable_file_path_with_sdm(sdm_mock):

    sdm_mock.return_value = sdm_mock
    sdm_model_path = MagicMock(name="sdm/file")
    sdm_model_path.__str__.return_value = "sdm/file"

    sdm_mock.get_calibration_table.return_value = sdm_model_path
    assert (
        get_gaintable_file_path(
            "output_dir", "filename", "sdm_path", "purpose", "field_id"
        )
        == "sdm/file"
    )

    sdm_mock.get_calibration_table.assert_called_once_with(
        field_id="field_id", purpose="purpose", file_name="filename"
    )
    sdm_model_path.parent.mkdir.assert_called_once_with(
        exists_ok=True, parents=True
    )
