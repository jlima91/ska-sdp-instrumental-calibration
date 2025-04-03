from mock import Mock

import ska_sdp_instrumental_calibration.workflow.stages.stage_3

stage_3 = ska_sdp_instrumental_calibration.workflow.stages.stage_3


def test_should_perform_stage_3():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = "config1"

    # act
    stage_3.stage_3.stage_definition(upstream_output, config1)

    # verify
    upstream_output.__setitem__.assert_called_once_with(
        "stage_3_data", ["config1"]
    )
