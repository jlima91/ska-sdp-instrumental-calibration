from mock import Mock, mock

from ska_sdp_instrumental_calibration.workflow.stages import load_data_stage


@mock.patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data.load_ms",
)
def test_should_load_data(load_ms_mock):
    load_ms_mock.return_value = "vis"

    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    fchunk = 1

    load_data_stage.stage_definition(
        upstream_output, fchunk, {"input": "path"}
    )

    load_ms_mock.assert_called_once_with("path", 1)
    upstream_output.__setitem__.assert_called_once_with("vis", "vis")
