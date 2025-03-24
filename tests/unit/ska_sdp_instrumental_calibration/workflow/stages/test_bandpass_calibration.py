from mock import Mock

import ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration

bp_cal = ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration


def test_should_bandpass_calibrate():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 123

    # act
    bp_cal.bandpass_calibration.stage_definition(upstream_output, config1)

    # verify
    upstream_output.__setitem__.assert_called_once_with(
        "bandpass_calibrated_data", [123]
    )
