import numpy as np
import xarray as xr
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.qa_metrics import QASection
from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.load_data import (
    Tags,
    load_data_stage,
)

# Stations of the MS: configuration names (as the gaintable antenna names)
# and their station IDs
STATION_NAMES = ["S8-1", "S8-2"]
STATION_IDS = ["0", "1"]


def _mock_vis(timestamps=(1.0, 2.0)):
    # The stage records vis.time.data.tolist() and the station
    # configuration in the run metadata, which validates them as
    # list[float] and dict[str, str], so a bare MagicMock is not enough
    vis = MagicMock(name="vis")
    vis.time.data = np.array(timestamps)
    vis.configuration.names.data = np.array(STATION_NAMES)
    vis.configuration.stations.data = np.array(STATION_IDS)
    return vis


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "load_data": {
            "nchannels_per_chunk": "auto",
            "ntimes_per_ms_chunk": 5,
            "cache_directory": None,
            "ack": False,
            "datacolumn": "DATA",
            "weightcolumn": "auto",
            "field_id": 0,
            "data_desc_id": 0,
        }
    }

    assert load_data_stage.__stage__.config == expected_config


def test_load_data_stage_is_a_broadcaster():
    assert load_data_stage in Tags.BROADCASTER


@patch("ska_sdp_instrumental_calibration.stages.load_data.get_derived_path")
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
    get_derived_path_mock,
):
    check_cache_mock.return_value = True
    read_data_mock.return_value = _mock_vis([10.0, 20.0, 30.0])

    gaintable = xr.DataArray(
        np.arange(12).reshape(1, 4, 3),
        dims=["time", "frequency", "antenna_name"],
    )
    create_bandpass_mock.return_value = gaintable
    read_ms_field_id_mock.return_value = "field-a"
    pre_flagging_data_path = (
        "/path/to/qa/dir/derived/vis/pre-flagging-data.json"
    )
    get_derived_path_mock.return_value = pre_flagging_data_path

    frequency_per_chunk = 2
    times_per_ms_chunk = 3

    upstream_output = UpstreamOutput()

    new_up_output = load_data_stage(
        upstream_output,
        "/path/to/output/dir",
        "/path/to/qa/dir",
        ["/path/to/vis.ms/"],
        frequency_per_chunk,
        times_per_ms_chunk,
        "/cache/dir/path",
        True,
        "DATA",
        "auto",
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
    assert new_up_output[0]["input_ms"] == "/path/to/vis.ms"

    get_derived_path_mock.assert_called_once_with(
        "/path/to/qa/dir", "vis/pre-flagging-data.json"
    )

    run_metadata = new_up_output[0]["run_metadata"]
    assert run_metadata.qa_dir == "/path/to/qa/dir"
    field_metadata = run_metadata.field_metadata["field-a"]
    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) == [
        "/path/to/vis.ms"
    ]
    assert field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.TIMESTAMPS
    ) == [[10.0, 20.0, 30.0]]
    # MS station ID keyed by station name, in configuration order
    assert field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.STATION_CONFIGURATION
    ) == {"S8-1": "0", "S8-2": "1"}
    assert field_metadata.get_data_for(QASection.FLAGGING.PRE_FLAGGING) == {
        "vis": pre_flagging_data_path
    }

    assert new_up_output[0]["chunks"] == {
        "time": -1,
        "baselineid": -1,
        "polarisation": -1,
        "spatial": -1,
        "frequency": frequency_per_chunk,
    }


@patch("ska_sdp_instrumental_calibration.stages.load_data.get_derived_path")
@patch("ska_sdp_instrumental_calibration.stages.load_data.os.makedirs")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".check_if_cache_files_exist"
)
@patch("ska_sdp_instrumental_calibration.stages.load_data.dask.compute")
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
    dask_compute_mock,
    check_cache_mock,
    os_makedirs_mock,
    get_derived_path_mock,
):
    check_cache_mock.return_value = False
    read_data_mock.return_value = _mock_vis()
    get_derived_path_mock.return_value = "/qa/derived/pre-flagging-data.json"

    gaintable = MagicMock(name="gaintable")
    create_bandpass_mock.return_value = gaintable
    read_ms_field_id_mock.return_value = "field-a"
    frequency_per_chunk = 16
    times_per_ms_chunk = 8

    upstream_output = UpstreamOutput()

    load_data_stage(
        upstream_output,
        "/path/to/output/dir",
        "/path/to/qa/dir",
        ["/path/to/subdir/../vis.ms/"],
        frequency_per_chunk,
        times_per_ms_chunk,
        None,
        False,
        "DATA",
        "auto",
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
        weightcolumn="auto",
        field_id=10,
        data_desc_id=5,
    )
    dask_compute_mock.assert_called_once_with(write_ms_mock.return_value)

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


@patch("ska_sdp_instrumental_calibration.stages.load_data.get_derived_path")
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
    get_derived_path_mock,
):
    check_cache_mock.return_value = True
    read_data_mock.return_value = _mock_vis()
    get_derived_path_mock.return_value = "/qa/derived/pre-flagging-data.json"
    read_vis_shape_mock.return_value = (10, 21, 64, 4)
    cluster_spec = MagicMock(name="cluster_spec")
    get_cluster_spec_mock.return_value = cluster_spec
    estimate_channels_mock.return_value = 8

    new_up_output = load_data_stage(
        UpstreamOutput(),
        "/path/to/output/dir",
        "/path/to/qa/dir",
        ["/path/to/vis.ms/"],
        "auto",
        5,
        "/cache/dir/path",
        False,
        "DATA",
        "auto",
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


@patch("ska_sdp_instrumental_calibration.stages.load_data.get_derived_path")
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
    get_derived_path_mock,
):
    check_cache_mock.return_value = False
    read_data_mock.return_value = _mock_vis()
    get_derived_path_mock.return_value = "/qa/derived/pre-flagging-data.json"
    template_vis = MagicMock(name="template_vis")
    template_vis.vis.shape = (10, 21, 64, 4)
    create_template_vis_mock.return_value = [template_vis]
    cluster_spec = MagicMock(name="cluster_spec")
    get_cluster_spec_mock.return_value = cluster_spec
    estimate_channels_mock.return_value = 8

    load_data_stage(
        UpstreamOutput(),
        "/path/to/output/dir",
        "/path/to/qa/dir",
        ["/path/to/vis.ms/"],
        "auto",
        5,
        "/cache/dir/path",
        True,
        "CORRECTED_DATA",
        "WEIGHT",
        2,
        3,
    )

    read_vis_shape_mock.assert_not_called()
    create_template_vis_mock.assert_called_once_with(
        "/path/to/vis.ms",
        ack=True,
        datacolumn="CORRECTED_DATA",
        weightcolumn="WEIGHT",
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
        weightcolumn="WEIGHT",
        field_id=2,
        data_desc_id=3,
    )


@patch("ska_sdp_instrumental_calibration.stages.load_data.get_derived_path")
@patch("ska_sdp_instrumental_calibration.stages.load_data.os.makedirs")
@patch(
    "ska_sdp_instrumental_calibration.stages.load_data"
    ".check_if_cache_files_exist"
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
def test_should_record_run_metadata_of_each_ms_of_a_field(
    read_ms_field_id_mock,
    create_bandpass_mock,
    read_data_mock,
    check_cache_mock,
    os_makedirs_mock,
    get_derived_path_mock,
):
    check_cache_mock.return_value = True
    read_ms_field_id_mock.return_value = "field-a"
    read_data_mock.side_effect = [
        _mock_vis([1.0, 2.0]),
        _mock_vis([3.0, 4.0]),
    ]
    get_derived_path_mock.side_effect = [
        "/qa/derived/vis_a/pre-flagging-data.json",
        "/qa/derived/vis_b/pre-flagging-data.json",
    ]

    new_up_output = load_data_stage(
        UpstreamOutput(),
        "/path/to/output/dir",
        "/qa",
        ["/path/to/vis_a.ms", "/path/to/vis_b.ms"],
        2,
        3,
        "/cache/dir/path",
        False,
        "DATA",
        "auto",
        0,
        0,
    )

    # Both MS share one run metadata, so the export stage writes all fields
    run_metadata = new_up_output[0]["run_metadata"]
    assert new_up_output[1]["run_metadata"] is run_metadata
    assert list(run_metadata.field_metadata) == ["field-a"]

    field_metadata = run_metadata.field_metadata["field-a"]
    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) == [
        "/path/to/vis_a.ms",
        "/path/to/vis_b.ms",
    ]
    assert field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.TIMESTAMPS
    ) == [[1.0, 2.0], [3.0, 4.0]]
    # MSs of a field share stations: recorded once for the field
    assert field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.STATION_CONFIGURATION
    ) == {"S8-1": "0", "S8-2": "1"}
    # Pre-flagging data is stored per MS prefix
    assert field_metadata.get_data_for(QASection.FLAGGING.PRE_FLAGGING) == {
        "vis_a": "/qa/derived/vis_a/pre-flagging-data.json",
        "vis_b": "/qa/derived/vis_b/pre-flagging-data.json",
    }

    get_derived_path_mock.assert_has_calls(
        [
            call("/qa", "vis_a/pre-flagging-data.json"),
            call("/qa", "vis_b/pre-flagging-data.json"),
        ]
    )
