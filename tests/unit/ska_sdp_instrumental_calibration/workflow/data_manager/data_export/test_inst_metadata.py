import pytest
from mock import Mock, call, patch
from ska_sdp_dataproduct_metadata import ObsCore

from ska_sdp_instrumental_calibration.data_managers.data_export.inst_metadata import (  # noqa: E501
    INSTMetaData,
)


@pytest.mark.parametrize(
    "eb,pb,expected",
    [
        ("eb_id", "pb_id", True),
        ("eb_id", None, False),
        (None, "pb_id", False),
        (None, None, False),
    ],
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers.data_export."
    "inst_metadata.os.environ.get"
)
def test_should_check_for_prerequisites(env_get_mock, eb, pb, expected):
    env_get_mock.side_effect = [eb, pb]
    assert INSTMetaData.can_create_metadata() == expected


@patch(
    "ska_sdp_instrumental_calibration.data_managers.data_export."
    "inst_metadata.os.path.exists",
    return_value=True,
)
def test_should_initialise_metadata_with_path(path_exists_mock):
    path = "/path/to/metadata.yml"
    with patch(
        "ska_sdp_instrumental_calibration.data_managers.data_export."
        "inst_metadata.MetaData"
    ) as metadata_mock:
        metadata_mock.return_value = metadata_mock
        path_exists_mock.return_value = False
        INSTMetaData(path)
        metadata_mock.assert_called_once_with()
        assert metadata_mock.output_path == path

    with patch(
        "ska_sdp_instrumental_calibration.data_managers.data_export."
        "inst_metadata.MetaData"
    ) as metadata_mock:
        metadata_mock.return_value = metadata_mock
        path_exists_mock.return_value = True
        INSTMetaData(path)
        metadata_mock.assert_called_once_with(path)
        assert metadata_mock.output_path == path


@patch(
    "ska_sdp_instrumental_calibration.data_managers.data_export."
    "inst_metadata.MetaData"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers.data_export."
    "inst_metadata.os.path.exists",
    return_value=True,
)
def test_should_export_metadata(path_exists_mock, metadata_mock, monkeypatch):
    """
    Test to check if required environment variables
        - EXECUTION_BLOCK_ID
        - PROCESSING_BLOCK_ID
    exist and then only export metadata.
    """
    mock_metadata_obj = Mock(name="metadata")
    data_mock = Mock(name="data")
    new_file_mock = Mock(name="new_file")
    mock_metadata_obj.new_file.return_value = new_file_mock
    data_mock.config = {"initial_config": "config"}
    mock_metadata_obj.get_data.return_value = data_mock
    metadata_mock.return_value = mock_metadata_obj
    inst_metadata = INSTMetaData("path/to/metadata")
    monkeypatch.setenv("EXECUTION_BLOCK_ID", "eb-id")
    monkeypatch.setenv("PROCESSING_BLOCK_ID", "pb-id")
    monkeypatch.setenv("PROCESSING_SCRIPT_IMAGE", "image")
    monkeypatch.setenv("PROCESSING_SCRIPT_NAME", "script")
    monkeypatch.setenv("PROCESSING_SCRIPT_VERSION", "version")

    inst_metadata = INSTMetaData(
        "metadata.yaml",
        [
            {"dp_path": "dp_path1", "description": "description1"},
            {"dp_path": "dp_path2", "description": "description2"},
        ],
    )
    inst_metadata.export().compute()

    mock_metadata_obj.set_execution_block_id.assert_called_once_with("eb-id")
    mock_metadata_obj.get_data.assert_called_once()
    assert data_mock.config == {
        "cmdline": None,
        "commit": None,
        "image": "image",
        "initial_config": "config",
        "processing_block": "pb-id",
        "processing_script": "script",
        "version": "version",
    }
    assert (
        data_mock.obscore.dataproduct_type == ObsCore.DataProductType.UNKNOWN
    )
    assert data_mock.obscore.calib_level == ObsCore.CalibrationLevel.LEVEL_0
    assert data_mock.obscore.obs_collection == (
        f"{ObsCore.SKA}/{ObsCore.SKA_LOW}/"
        f"{ObsCore.DataProductType.UNKNOWN}"
    )

    assert data_mock.obscore.access_format == ObsCore.AccessFormat.UNKNOWN
    assert data_mock.obscore.facility_name == ObsCore.SKA
    assert data_mock.obscore.instrument_name == ObsCore.SKA_LOW

    mock_metadata_obj.new_file.assert_has_calls(
        [
            call(dp_path="dp_path1", description="description1"),
            call(dp_path="dp_path2", description="description2"),
        ]
    )

    new_file_mock.update_status.assert_has_calls([call("done"), call("done")])

    mock_metadata_obj.write.assert_called_once()
