from mock import Mock

from ska_sdp_instrumental_calibration.workflow.stages import delay_calibration


def test_should_delay_calibrate():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 123

    # act
    delay_calibration.delay_calibration.stage_definition(
        upstream_output, config1
    )

    # verify
    upstream_output.__setitem__.assert_called_once_with(
        "delay_calibrated_data", [123]
    )
