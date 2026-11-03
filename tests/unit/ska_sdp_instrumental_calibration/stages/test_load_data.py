import numpy as np
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.load_data import (
    Tags,
    load_data_stage,
)


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "load_data": {
            "nchannels_per_chunk": "auto",
            "ntimes_per_ms_chunk": 5,
            "cache_directory": None,
            "ack": False,
            "datacolumn": "DATA",
            "field_id": 0,
            "data_desc_id": 0,
        }
    }

    assert load_data_stage.__stage__.config == expected_config


def test_load_data_stage_is_a_broadcaster():
    assert load_data_stage in Tags.BROADCASTER


@patch("ska_sdp_instrumental_calibration.stages.load_data.os.makedirs")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.write_ms_to_zarr")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".read_visibility_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".create_gaintable_from_visibility"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.read_ms_field_id")
def test_should_load_data_from_existing_cached_zarr_file(
    read_ms_field_id_mock,
    create_bandpass_mock,
    read_data_mock,
    write_ms_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = True

    gaintable = xr.DataArray(
        np.arange(12).reshape(1, 4, 3),
        dims=["time", "frequency", "antenna_name"],
    )
    create_bandpass_mock.return_value = gaintable
    read_ms_field_id_mock.return_value = "field-a"

    frequency_per_chunk = 2
    times_per_ms_chunk = 3

    upstream_output = UpstreamOutput()

    new_up_output = load_data_stage(
        upstream_output,
        "/path/to/output/dir",
        ["/path/to/vis.ms/"],
        frequency_per_chunk,
        times_per_ms_chunk,
        "/cache/dir/path",
        True,
        "DATA",
        2,
        4,
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
            "time": -1,
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "frequency": frequency_per_chunk,
        },
    )

    create_bandpass_mock.assert_called_once_with(
        read_data_mock.return_value, "full", "B"
    )

    assert len(new_up_output) == 1
    assert new_up_output[0]["vis"] == read_data_mock.return_value
    assert new_up_output[0]["ms_prefix"] == "vis"
    assert new_up_output[0]["central_beams"] is None
    assert new_up_output[0]["beams_factory"] is None
    assert new_up_output[0]["field_id"] == "field-a"
    assert new_up_output[0]["calibration_purpose"] == "bandpass"

    assert new_up_output[0]["chunks"] == {
        "time": -1,
        "baselineid": -1,
        "polarisation": -1,
        "spatial": -1,
        "frequency": frequency_per_chunk,
    }


@patch("ska_sdp_instrumental_calibration.stages.load_data.os.makedirs")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.write_ms_to_zarr")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".read_visibility_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".create_gaintable_from_visibility"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.read_ms_field_id")
def test_should_write_ms_if_zarr_is_not_cached_and_load_from_zarr(
    read_ms_field_id_mock,
    create_bandpass_mock,
    read_data_mock,
    write_ms_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = False

    gaintable = MagicMock(name="gaintable")
    create_bandpass_mock.return_value = gaintable
    read_ms_field_id_mock.return_value = "field-a"
    frequency_per_chunk = 16
    times_per_ms_chunk = 8

    upstream_output = UpstreamOutput()

    load_data_stage(
        upstream_output,
        "/path/to/output/dir",
        ["/path/to/subdir/../vis.ms/"],
        frequency_per_chunk,
        times_per_ms_chunk,
        None,
        False,
        "DATA",
        10,
        5,
    )

    os_makedirs_mock.assert_called_once_with(
        "/path/to/output/dir/.cache/vis.ms_fid10_ddid5",
        mode=0o755,
        exist_ok=True,
    )

    check_cache_mock.assert_called_once_with(
        "/path/to/output/dir/.cache/vis.ms_fid10_ddid5"
    )

    write_ms_mock.assert_called_once_with(
        "/path/to/vis.ms",
        "/path/to/output/dir/.cache/vis.ms_fid10_ddid5",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": times_per_ms_chunk,
            "frequency": frequency_per_chunk,
        },
        ack=False,
        datacolumn="DATA",
        field_id=10,
        data_desc_id=5,
    )

    read_data_mock.assert_called_once_with(
        "/path/to/output/dir/.cache/vis.ms_fid10_ddid5",
        {
            "time": -1,
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "frequency": frequency_per_chunk,
        },
    )


@patch("ska_sdp_instrumental_calibration.stages.load_data.os.makedirs")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".read_vis_shape_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".create_template_vis_from_ms"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.get_cluster_spec")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".estimate_channels_per_chunk"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".read_visibility_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".create_gaintable_from_visibility"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.read_ms_field_id")
def test_should_estimate_frequency_chunks_from_cached_zarr_shape(
    read_ms_field_id_mock,
    create_bandpass_mock,
    read_data_mock,
    estimate_channels_mock,
    get_cluster_spec_mock,
    create_template_vis_mock,
    read_vis_shape_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = True
    read_vis_shape_mock.return_value = (10, 21, 64, 4)
    cluster_spec = MagicMock(name="cluster_spec")
    get_cluster_spec_mock.return_value = cluster_spec
    estimate_channels_mock.return_value = 8

    new_up_output = load_data_stage(
        UpstreamOutput(),
        "/path/to/output/dir",
        ["/path/to/vis.ms/"],
        "auto",
        5,
        "/cache/dir/path",
        False,
        "DATA",
        0,
        0,
    )

    read_vis_shape_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_fid0_ddid0"
    )
    create_template_vis_mock.assert_not_called()
    estimate_channels_mock.assert_called_once_with(
        64, 10 * 21 * 4, cluster_spec
    )
    read_data_mock.assert_called_once_with(
        "/cache/dir/path/vis.ms_fid0_ddid0",
        {
            "time": -1,
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "frequency": 8,
        },
    )
    assert new_up_output[0]["chunks"]["frequency"] == 8


@patch("ska_sdp_instrumental_calibration.stages.load_data.os.makedirs")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".read_vis_shape_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".create_template_vis_from_ms"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.get_cluster_spec")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".estimate_channels_per_chunk"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.write_ms_to_zarr")
@patch("ska_sdp_instrumental_calibration.stages.load_data.dask.compute")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".read_visibility_from_zarr"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".create_gaintable_from_visibility"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.read_ms_field_id")
def test_should_estimate_frequency_chunks_from_ms_before_writing_zarr(
    read_ms_field_id_mock,
    create_bandpass_mock,
    read_data_mock,
    dask_compute_mock,
    write_ms_mock,
    estimate_channels_mock,
    get_cluster_spec_mock,
    create_template_vis_mock,
    read_vis_shape_mock,
    check_cache_mock,
    os_makedirs_mock,
):
    check_cache_mock.return_value = False
    template_vis = MagicMock(name="template_vis")
    template_vis.vis.shape = (10, 21, 64, 4)
    create_template_vis_mock.return_value = [template_vis]
    cluster_spec = MagicMock(name="cluster_spec")
    get_cluster_spec_mock.return_value = cluster_spec
    estimate_channels_mock.return_value = 8

    load_data_stage(
        UpstreamOutput(),
        "/path/to/output/dir",
        ["/path/to/vis.ms/"],
        "auto",
        5,
        "/cache/dir/path",
        True,
        "CORRECTED_DATA",
        2,
        3,
    )

    read_vis_shape_mock.assert_not_called()
    create_template_vis_mock.assert_called_once_with(
        "/path/to/vis.ms",
        ack=True,
        datacolumn="CORRECTED_DATA",
        field_ids=[2],
        data_desc_ids=[3],
    )
    estimate_channels_mock.assert_called_once_with(
        64, 10 * 21 * 4, cluster_spec
    )
    write_ms_mock.assert_called_once_with(
        "/path/to/vis.ms",
        "/cache/dir/path/vis.ms_fid2_ddid3",
        {
            "baselineid": -1,
            "polarisation": -1,
            "spatial": -1,
            "time": 5,
            "frequency": 8,
        },
        ack=True,
        datacolumn="CORRECTED_DATA",
        field_id=2,
        data_desc_id=3,
    )
