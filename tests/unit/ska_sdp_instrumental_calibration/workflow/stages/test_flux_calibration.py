from mock import Mock

from ska_sdp_instrumental_calibration.workflow.stages.flux_calibration import (
    flux_calibration,
)


def test_should_flux_calibrate():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 123

    # act
    flux_calibration.stage_definition(upstream_output, config1)

    # verify
    upstream_output.__setitem__.assert_called_once_with(
        "flux_calibrated_data", [123]
    )
