from mock import Mock, mock

from ska_sdp_instrumental_calibration.workflow.stages import load_data_stage


@mock.patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data."
    "create_bandpass_table",
)
@mock.patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data.load_ms",
)
def test_should_load_data(load_ms_mock, create_bandpass_table_mock):
    load_ms_mock.return_value = "vis"
    gaintable_mock = Mock(name="gaintable")
    create_bandpass_table_mock.return_value = gaintable_mock

    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    fchunk = 1
    ack = False
    start_chan = 0
    end_chan = 5
    datacolumn = "DATA"
    selected_sources = ["S1"]
    selected_dds = ["D1"]
    average_channels = False
    gaintable_mock.chunk.return_value = gaintable_mock

    load_data_stage.stage_definition(
        upstream_output,
        fchunk,
        ack,
        start_chan,
        end_chan,
        datacolumn,
        selected_sources,
        selected_dds,
        average_channels,
        {"input": "path"},
    )

    load_ms_mock.assert_called_once_with(
        "path",
        1,
        ack=False,
        start_chan=0,
        end_chan=5,
        datacolumn="DATA",
        selected_sources=["S1"],
        selected_dds=["D1"],
        average_channels=False,
    )
    create_bandpass_table_mock.assert_called_once_with("vis")
    gaintable_mock.chunk.assert_called_once_with({"frequency": 1})
    upstream_output.__setitem__.assert_has_calls(
        [
            mock.call("vis", "vis"),
            mock.call("corrected_vis", "vis"),
            mock.call("gaintable", gaintable_mock),
            mock.call("beams", None),
        ]
    )
