from mock import patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import target_calibration

load_data_stage = target_calibration.load_data_stage


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.os.makedirs"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.check_if_cache_files_exist"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.write_ms_to_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.read_visibility_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.create_gaintable_from_visibility"
)
def test_should_load_data_from_existing_cached_zarr_file(
    create_gaintable_mock,
    read_data_mock,
    write_ms_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = True

    frequency_per_chunk = 2
    times_per_ms_chunk = 3
    timeslice = 1

    upstream_output = UpstreamOutput()

    new_up_output = load_data_stage.stage_definition(
        upstream_output,
        frequency_per_chunk,
        times_per_ms_chunk,
        "/cache/dir/path",
        timeslice,
        True,
        "ANOTHER_DATA",
        2,
        4,
        {"input": "/path/to/vis.ms/"},
        "/path/to/output/dir",
    )

    os_makedirs_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_fid2_ddid4", mode=0o755, exist_ok=True
    )

    check_cache_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_fid2_ddid4"
    )

    write_ms_mock.assert_not_called()

    read_data_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_fid2_ddid4",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": times_per_ms_chunk,
            "frequency": -1,
        },
    )

    create_gaintable_mock.assert_called_once_with(
        read_data_mock.return_value, timeslice, "G"
    )

    assert new_up_output["vis"] == read_data_mock.return_value
    assert new_up_output["timeslice"] == 1
    assert new_up_output["gaintable"] == create_gaintable_mock.return_value
    assert new_up_output["central_beams"] is None


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.os.makedirs"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.check_if_cache_files_exist"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.write_ms_to_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.read_visibility_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".load_data.create_gaintable_from_visibility"
)
def test_should_write_ms_if_zarr_is_not_cached_and_load_from_zarr(
    create_gaintable_mock,
    read_data_mock,
    write_ms_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = False

    frequency_per_chunk = 16
    times_per_ms_chunk = 8

    upstream_output = UpstreamOutput()

    load_data_stage.stage_definition(
        upstream_output,
        frequency_per_chunk,
        times_per_ms_chunk,
        None,
        0,
        False,
        "ANOTHER_DATA",
        10,
        5,
        {"input": "/path/to/subdir/../vis.ms/"},
        "/path/to/output/dir",
    )

    os_makedirs_mock.assert_called_once_with(
        "/path/to/output/dir/vis.ms_fid10_ddid5", mode=0o755, exist_ok=True
    )

    check_cache_mock.assert_called_once_with(
        "/path/to/output/dir/vis.ms_fid10_ddid5"
    )

    write_ms_mock.assert_called_once_with(
        "/path/to/vis.ms",
        "/path/to/output/dir/vis.ms_fid10_ddid5",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": times_per_ms_chunk,
            "frequency": frequency_per_chunk,
        },
        ack=False,
        datacolumn="ANOTHER_DATA",
        field_id=10,
        data_desc_id=5,
    )

    read_data_mock.assert_called_once_with(
        "/path/to/output/dir/vis.ms_fid10_ddid5",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": times_per_ms_chunk,
            "frequency": -1,
        },
    )
