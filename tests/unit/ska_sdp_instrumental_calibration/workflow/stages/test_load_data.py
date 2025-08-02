import numpy as np
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages.load_data import (
    load_data_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data" ".os.makedirs"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".write_ms_to_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".read_dataset_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".create_bandpass_table"
)
def test_should_load_data_from_existing_cached_zarr_file(
    create_bandpass_mock,
    read_data_mock,
    write_ms_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = True

    gaintable = xr.DataArray(
        np.arange(12).reshape(1, 4, 3), dims=["time", "frequency", "antenna"]
    )
    create_bandpass_mock.return_value = gaintable

    frequency_per_chunk = 2
    times_per_ms_chunk = 3

    upstream_output = UpstreamOutput()

    new_up_output = load_data_stage.stage_definition(
        upstream_output,
        frequency_per_chunk,
        times_per_ms_chunk,
        "/cache/dir/path",
        True,
        "ANOTHER_DATA",
        2,
        4,
        {"input": "/path/to/vis.ms"},
    )

    os_makedirs_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_2_4", mode=0o755, exist_ok=True
    )

    check_cache_mock.assert_called_once_with("/cache/dir/path/vis.ms_2_4")

    write_ms_mock.assert_not_called()

    read_data_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_2_4",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": -1,
            "frequency": frequency_per_chunk,
        },
    )

    create_bandpass_mock.assert_called_once_with(read_data_mock.return_value)

    assert new_up_output["vis"] == read_data_mock.return_value
    assert new_up_output["beams"] is None

    assert dict(new_up_output["gaintable"].chunksizes) == {
        "time": (1,),
        "frequency": (
            2,
            2,
        ),
        "antenna": (3,),
    }


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data" ".os.makedirs"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".write_ms_to_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".read_dataset_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data"
    ".create_bandpass_table"
)
def test_should_write_ms_if_zarr_is_not_cached_and_load_from_zarr(
    create_bandpass_mock,
    read_data_mock,
    write_ms_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = False

    gaintable = MagicMock(name="gaintable")
    create_bandpass_mock.return_value = gaintable

    frequency_per_chunk = 16
    times_per_ms_chunk = 8

    upstream_output = UpstreamOutput()

    load_data_stage.stage_definition(
        upstream_output,
        frequency_per_chunk,
        times_per_ms_chunk,
        "/cache/dir/path",
        True,
        "ANOTHER_DATA",
        10,
        5,
        {"input": "/path/to/vis.ms"},
    )

    os_makedirs_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_10_5", mode=0o755, exist_ok=True
    )

    check_cache_mock.assert_called_once_with("/cache/dir/path/vis.ms_10_5")

    write_ms_mock.assert_called_once_with(
        "/path/to/vis.ms",
        "/cache/dir/path/vis.ms_10_5",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": times_per_ms_chunk,
            "frequency": frequency_per_chunk,
        },
        ack=True,
        datacolumn="ANOTHER_DATA",
        field_id=10,
        data_desc_id=5,
    )

    read_data_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_10_5",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": -1,
            "frequency": frequency_per_chunk,
        },
    )
