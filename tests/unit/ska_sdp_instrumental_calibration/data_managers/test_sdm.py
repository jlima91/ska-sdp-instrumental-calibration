from pathlib import Path

import pytest
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.sdm import (
    get_gaintable_file_path,
    get_qa_metric_dir,
    get_qa_metric_field_id,
    prepare_qa_path,
)


def test_should_prepare_qa_path_when_sdm_path_exists(tmp_path):
    sdm_root = tmp_path / "sdm"
    logs = sdm_root / "logs"
    logs.mkdir(parents=True)
    expected_inst_log_path = Path(f"{sdm_root}/logs/01-inst")

    assert expected_inst_log_path.exists() is False
    assert prepare_qa_path(None, sdm_path=sdm_root) == str(
        expected_inst_log_path
    )
    assert expected_inst_log_path.exists() is True


def test_should_return_output_dir_as_qa_path_when_sdm_path_is_none():
    assert prepare_qa_path("output_dir", sdm_path=None) == "output_dir"


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
        exist_ok=True, parents=True
    )


def test_should_return_output_path_as_qa_metric_dir_when_sdm_path_is_none():
    assert get_qa_metric_dir("output_dir", sdm_path=None) == "output_dir"


def test_should_return_latest_inst_logs_dir_as_qa_metric_dir(tmp_path):
    logs = tmp_path / "logs"
    (logs / "01-inst").mkdir(parents=True)
    (logs / "02-inst").mkdir()

    assert get_qa_metric_dir(None, sdm_path=str(tmp_path)) == str(
        logs / "02-inst"
    )


def test_should_raise_error_when_sdm_path_for_qa_metric_does_not_exist():
    with pytest.raises(FileNotFoundError):
        get_qa_metric_dir(None, sdm_path="non_existent_sdm_path")


def test_should_get_qa_metric_field_id_from_logs_dir_with_sdm():
    assert get_qa_metric_field_id("sdm/logs/02-inst", sdm_path="sdm") == "02"


def test_should_get_qa_metric_field_id_from_gaintable_without_sdm(tmp_path):
    (tmp_path / "field_id_gaintable.h5parm").touch()

    assert get_qa_metric_field_id(str(tmp_path)) == "field_id"


def test_should_raise_error_when_no_gaintable_for_qa_metric_field_id(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        get_qa_metric_field_id(str(tmp_path))
