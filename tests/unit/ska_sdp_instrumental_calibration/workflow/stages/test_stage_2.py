from mock import Mock

from ska_sdp_instrumental_calibration.workflow.stages.stage_2 import stage_2


def test_should_perform_stage_2():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 1

    # act
    stage_2.stage_definition(upstream_output, config1)

    # verify
    upstream_output.__setitem__.assert_called_once_with("stage_2_data", [1])
