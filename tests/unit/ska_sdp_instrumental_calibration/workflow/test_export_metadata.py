from mock import Mock, patch

from ska_sdp_instrumental_calibration.workflow.export_metadata import (
    export_metadata,
    export_metadata_file,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.export_metadata.export_metadata"
)
def test_should_export_metadata(export_metadata_mock, monkeypatch):
    """
    Test to check if required environment variables
        - EXECUTION_BLOCK
        - PROCESSING_BLOCK
    exist and then only export metadata.
    """
    monkeypatch.setenv("EXECUTION_BLOCK", "eb-id")
    monkeypatch.setenv("PROCESSING_BLOCK", "pb-id")
    monkeypatch.setenv("IMAGE", "image")
    monkeypatch.setenv("PROCESSING_SCRIPT", "script")
    monkeypatch.setenv("SDP_SCRIPT_VERSION", "version")

    export_metadata_file("metadata.yaml")

    export_metadata_mock.assert_called_once_with(
        "metadata.yaml",
        "eb-id",
        {
            "cmdline": None,
            "commit": None,
            "image": "image",
            "processing_block": "pb-id",
            "processing_script": "script",
            "version": "version",
        },
        [],
    )


@patch("ska_sdp_instrumental_calibration.workflow.export_metadata.dask")
def test_should_not_export_metadata(dask_mock):
    """
    Test to check if required environment variables
        - EXECUTION_BLOCK
        - PROCESSING_BLOCK
    do not exist and does not create metadata file.
    """
    export_metadata_file("metadata.yaml")

    dask_mock.delayed.assert_called_once_with(None)


@patch("ska_sdp_instrumental_calibration.workflow.export_metadata.os")
@patch("ska_sdp_instrumental_calibration.workflow.export_metadata.MetaData")
def test_should_export_new_metadata_with_data_products(metadata_mock, os_mock):
    """
    Test to export metadata with data products information.
    """
    os_mock.path.exists.return_value = False

    metadata_data_mock = Mock(name="data")
    metadata_data_mock.config = {}
    metadata_mock.get_data.return_value = metadata_data_mock

    metadata_file_mock = Mock(name="file")
    metadata_mock.new_file.return_value = metadata_file_mock

    metadata_mock.return_value = metadata_mock

    export_metadata(
        "metadata.yaml",
        "eb-id",
        {},
        data_products=[{"dp_path": "file.dat", "description": "data table"}],
    ).compute()

    os_mock.path.exists.assert_called_once_with("metadata.yaml")
    metadata_mock.assert_called_once()

    assert metadata_mock.output_path == "metadata.yaml"

    metadata_mock.set_execution_block_id.assert_called_once_with("eb-id")
    metadata_mock.get_data.assert_called_once()
    metadata_mock.new_file.assert_called_once_with(
        dp_path="file.dat", description="data table"
    )

    metadata_file_mock.update_status.assert_called_once_with("done")

    metadata_mock.write.assert_called_once()


@patch("ska_sdp_instrumental_calibration.workflow.export_metadata.os")
@patch("ska_sdp_instrumental_calibration.workflow.export_metadata.MetaData")
def test_should_export_existing_metadata_with_data_products(
    metadata_mock, os_mock
):
    """
    Test to export exiting metadata with data products information.
    """
    os_mock.path.exists.return_value = True

    metadata_data_mock = Mock(name="data")
    metadata_data_mock.config = {}
    metadata_mock.get_data.return_value = metadata_data_mock

    metadata_file_mock = Mock(name="file")
    metadata_mock.new_file.return_value = metadata_file_mock

    metadata_mock.return_value = metadata_mock

    export_metadata(
        "metadata.yaml",
        "eb-id",
        {},
        data_products=[{"dp_path": "file.dat", "description": "data table"}],
    ).compute()

    os_mock.path.exists.assert_called_once_with("metadata.yaml")
    metadata_mock.assert_called_once_with(path="metadata.yaml")

    assert metadata_mock.output_path == "metadata.yaml"

    metadata_mock.set_execution_block_id.assert_called_once_with("eb-id")
    metadata_mock.get_data.assert_called_once()
    metadata_mock.new_file.assert_called_once_with(
        dp_path="file.dat", description="data table"
    )

    metadata_file_mock.update_status.assert_called_once_with("done")

    metadata_mock.write.assert_called_once()
