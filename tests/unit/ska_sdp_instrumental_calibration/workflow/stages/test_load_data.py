from mock import Mock, mock

from ska_sdp_instrumental_calibration.workflow.stages.load_data import (
    load_data,
)


@mock.patch(
    "ska_sdp_instrumental_calibration.workflow.stages.load_data.load_ms"
)
def test_should_load_data(load_ms_mock):
    # mocks setup
    load_ms_mock.return_value = "vis"

    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    fchunk = 1

    # act
    load_data.stage_definition(upstream_output, fchunk, {"input": "path"})

    # verify
    load_ms_mock.assert_called_once_with("path", 1)
    upstream_output.__setitem__.assert_called_once_with("vis", "vis")
