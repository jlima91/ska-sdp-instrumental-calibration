from mock import Mock

from ska_sdp_instrumental_calibration.workflow.stages.flag_ms import flag_ms


def test_should_flag_ms():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 1

    # act
    flag_ms.stage_definition(upstream_output, config1)

    # verify
    upstream_output.__setitem__.assert_called_once_with("flagged_data", [1])
