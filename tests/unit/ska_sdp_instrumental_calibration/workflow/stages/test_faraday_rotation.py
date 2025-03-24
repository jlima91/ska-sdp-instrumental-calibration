from mock import Mock

from ska_sdp_instrumental_calibration.workflow.stages.faraday_rotation import (
    faraday_rotation,
)


def test_should_faraday_rotate():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 123

    # act
    faraday_rotation.stage_definition(upstream_output, config1)

    # verify
    upstream_output.__setitem__.assert_called_once_with(
        "faraday_rotation_result", [123]
    )
