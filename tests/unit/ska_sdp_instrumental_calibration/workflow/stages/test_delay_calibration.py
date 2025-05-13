from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    delay_calibration_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_perform_delay_calibration(apply_delay_mock):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    oversample = 16
    plot_config = {"plot_table": False, "fixed_axis": False}

    actual_output = delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        _output_dir_="/output/path",
    )

    apply_delay_mock.assert_called_once_with(gaintable_mock, oversample)

    assert actual_output.gaintable == apply_delay_mock.return_value


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_plot_the_delayed_gaintable_with_proper_suffix(
    apply_delay_mock, plot_gaintable_mock
):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    oversample = 16
    plot_config = {"plot_table": True, "fixed_axis": True}

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        _output_dir_="/output/path",
    )

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        _output_dir_="/output/path",
    )

    plot_gaintable_mock.assert_has_calls(
        [
            call(
                apply_delay_mock.return_value,
                "/output/path/delay",
                figure_title="Delay",
                fixed_axis=True,
            ),
            call(
                apply_delay_mock.return_value,
                "/output/path/delay_1",
                figure_title="Delay",
                fixed_axis=True,
            ),
        ]
    )
