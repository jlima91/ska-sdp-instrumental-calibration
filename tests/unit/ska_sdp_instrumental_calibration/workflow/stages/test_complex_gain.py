from mock import Mock

from ska_sdp_instrumental_calibration.workflow.stages.complex_gain import (
    complex_gain,
)


def test_should_do_complex_gain():
    # init
    upstream_output = Mock(name="upstream_output")
    upstream_output.__setitem__ = Mock(name="upstream-output-setitem")
    config1 = 123
    config2 = 1234

    # act
    complex_gain.stage_definition(upstream_output, config1, config2)

    # verify
    upstream_output.__setitem__.assert_called_once_with(
        "complex_gains", [123, 1234]
    )
