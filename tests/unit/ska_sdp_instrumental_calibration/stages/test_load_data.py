import pytest
from mock import MagicMock, patch
from pydantic import ValidationError

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.load_data import (
    Tags,
    load_data_stage,
)

MODULE = "ska_sdp_instrumental_calibration.stages.load_data"


def _run_stage(inputs, **overrides):
    """Run load_data_stage with default config, overriding some values."""
    config = {
        "field_id": 0,
        "ddi_id": 0,
        "field_name": None,
        "spw_name": None,
        "scan_intents": None,
        "data_group": "base",
        "nchannels_per_chunk": 8,
    }
    config.update(overrides)
    return load_data_stage(
        UpstreamOutput(), "/output/dir", inputs, *config.values()
    )


@pytest.fixture
def mocks():
    """Patch every collaborator of load_data_stage."""
    names = [
        "detect_input_type",
        "convert_msv2_to_processing_set",
        "read_msv2_selection_shape",
        "open_msv4_selection",
        "chunk_ms_xdt",
        "get_field_name",
        "create_gaintable_from_msv4",
        "estimate_channels_per_chunk",
        "get_cluster_spec",
    ]
    patchers = {name: patch(f"{MODULE}.{name}") for name in names}
    started = {name: patcher.start() for name, patcher in patchers.items()}
    started["get_field_name"].return_value = "CAL_FIELD_0"

    yield started

    for patcher in patchers.values():
        patcher.stop()


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "load_data": {
            "field_id": 0,
            "ddi_id": 0,
            "field_name": None,
            "spw_name": None,
            "scan_intents": None,
            "data_group": "base",
            "nchannels_per_chunk": "auto",
        }
    }

    assert load_data_stage.__stage__.config == expected_config


def test_load_data_stage_is_a_broadcaster():
    assert load_data_stage in Tags.BROADCASTER


def test_should_convert_each_msv2_with_pipeline_chunks(mocks):
    mocks["detect_input_type"].return_value = "msv2"

    outputs = _run_stage(["/data/a.ms", "/data/b.ms/"])

    mocks["detect_input_type"].assert_called_once_with(
        ["/data/a.ms", "/data/b.ms"]
    )
    converted = [
        call.kwargs
        for call in mocks["convert_msv2_to_processing_set"].call_args_list
    ]
    assert [kwargs["in_file"] for kwargs in converted] == [
        "/data/a.ms",
        "/data/b.ms",
    ]
    assert [kwargs["out_file"] for kwargs in converted] == [
        "/output/dir/processing_sets/a.ps.zarr",
        "/output/dir/processing_sets/b.ps.zarr",
    ]
    for kwargs in converted:
        assert kwargs["partition_scheme"] == ["FIELD_ID"]
        assert kwargs["main_chunksize"] == {"frequency": 8}
        assert kwargs["persistence_mode"] == "w"
    assert [output["ms_prefix"] for output in outputs] == ["a", "b"]


@pytest.mark.parametrize(
    "partition, selected",
    [
        ({"FIELD_ID": [2], "DATA_DESC_ID": [3]}, True),
        ({"FIELD_ID": [0], "DATA_DESC_ID": [3]}, False),
        ({"FIELD_ID": [2], "DATA_DESC_ID": [0]}, False),
    ],
)
def test_should_convert_only_the_selected_field_and_ddi(
    mocks, partition, selected
):
    mocks["detect_input_type"].return_value = "msv2"

    _run_stage(["/data/a.ms"], field_id=2, ddi_id=3)

    partition_filter = mocks[
        "convert_msv2_to_processing_set"
    ].call_args.kwargs["partition_filter"]
    assert partition_filter(partition) is selected


def test_should_ignore_names_for_msv2_inputs(mocks):
    mocks["detect_input_type"].return_value = "msv2"

    _run_stage(["/data/a.ms"], field_name="F", spw_name="S")

    # The converted Processing Set holds the selected field and DDI only
    mocks["open_msv4_selection"].assert_called_once_with(
        "/output/dir/processing_sets/a.ps.zarr",
        field_name=None,
        spw_name=None,
        scan_intents=None,
        data_group="base",
    )


def test_should_read_processing_sets_in_place_with_name_selection(mocks):
    mocks["detect_input_type"].return_value = "ps"

    outputs = _run_stage(
        ["/data/obs.ps.zarr"],
        field_id=4,
        field_name="CAL_FIELD_0",
        spw_name="spw_0",
        scan_intents=["CALIBRATE_BANDPASS#ON_SOURCE"],
        data_group="corrected",
    )

    mocks["convert_msv2_to_processing_set"].assert_not_called()
    mocks["open_msv4_selection"].assert_called_once_with(
        "/data/obs.ps.zarr",
        field_name="CAL_FIELD_0",
        spw_name="spw_0",
        scan_intents=["CALIBRATE_BANDPASS#ON_SOURCE"],
        data_group="corrected",
    )
    assert outputs[0]["ms_prefix"] == "obs"


def test_should_fill_upstream_output(mocks):
    mocks["detect_input_type"].return_value = "ps"

    (output,) = _run_stage(["/data/obs.ps.zarr"])

    mocks["chunk_ms_xdt"].assert_called_once_with(
        mocks["open_msv4_selection"].return_value, 8
    )
    ms_xdt = mocks["chunk_ms_xdt"].return_value
    mocks["create_gaintable_from_msv4"].assert_called_once_with(
        ms_xdt, "full", "B"
    )
    mocks["get_field_name"].assert_called_once_with(ms_xdt)

    assert output["vis"] == ms_xdt
    assert output["gaintable"] == (
        mocks["create_gaintable_from_msv4"].return_value
    )
    # Without the "_<FIELD_ID>" suffix added by xradio
    assert output["field_id"] == "CAL_FIELD"
    assert output["central_beams"] is None
    assert output["beams_factory"] is None
    assert output["calibration_purpose"] == "bandpass"


def test_should_raise_for_inputs_with_the_same_prefix(mocks):
    mocks["detect_input_type"].return_value = "msv2"

    with pytest.raises(ValueError, match=r"duplicates: \['obs'\]"):
        _run_stage(["/night1/obs.ms", "/night2/obs.ms"])

    mocks["convert_msv2_to_processing_set"].assert_not_called()


def test_should_estimate_frequency_chunks_from_msv2_before_converting(
    mocks,
):
    mocks["detect_input_type"].return_value = "msv2"
    mocks["read_msv2_selection_shape"].return_value = (10, 21, 64, 4)
    mocks["estimate_channels_per_chunk"].return_value = 16

    _run_stage(
        ["/data/a.ms"], field_id=1, ddi_id=2, nchannels_per_chunk="auto"
    )

    mocks["read_msv2_selection_shape"].assert_called_once_with(
        "/data/a.ms", 1, 2
    )
    mocks["estimate_channels_per_chunk"].assert_called_once_with(
        64, 10 * 21 * 4, mocks["get_cluster_spec"].return_value
    )
    assert mocks["convert_msv2_to_processing_set"].call_args.kwargs[
        "main_chunksize"
    ] == {"frequency": 16}
    mocks["chunk_ms_xdt"].assert_called_once_with(
        mocks["open_msv4_selection"].return_value, 16
    )


def test_should_estimate_frequency_chunks_from_processing_set(mocks):
    mocks["detect_input_type"].return_value = "ps"
    mocks["estimate_channels_per_chunk"].return_value = 16
    ms_xdt = mocks["open_msv4_selection"].return_value
    ms_xdt.attrs = {"data_groups": {"corrected": {"correlated_data": "VC"}}}
    ms_xdt.ds = {"VC": MagicMock(shape=(10, 21, 64, 4))}

    _run_stage(["/data/obs.ps.zarr"], nchannels_per_chunk="auto")

    mocks["read_msv2_selection_shape"].assert_not_called()
    mocks["estimate_channels_per_chunk"].assert_called_once_with(
        64, 10 * 21 * 4, mocks["get_cluster_spec"].return_value
    )
    mocks["chunk_ms_xdt"].assert_called_once_with(ms_xdt, 16)


@pytest.mark.parametrize("data_group", ["base", "corrected", "model"])
def test_should_accept_data_groups_converted_from_msv2(data_group):
    params = load_data_stage.__stage__.metadata.stage_param(
        data_group=data_group
    )

    assert params.data_group == data_group


def test_should_reject_other_data_groups():
    with pytest.raises(
        ValidationError, match="'base', 'corrected' or 'model'"
    ):
        load_data_stage.__stage__.metadata.stage_param(data_group="imaging")
