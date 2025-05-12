from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    smooth_gain_solution_stage,
)


def test_should_smooth_the_gian_solution():
    upstream_output = UpstreamOutput()
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")

    rolled_array_mock.median.return_value = smooth_gain_mock
    gaintable_mock.gain.rolling.return_value = rolled_array_mock
    upstream_output.gaintable = gaintable_mock

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "median", False, "./output/path"
    )

    gaintable_mock.gain.rolling.assert_called_once_with(
        frequency=3, center=True
    )
    rolled_array_mock.median.assert_called_once_with()
    gaintable_mock.assign.assert_called_once_with({"gain": smooth_gain_mock})


def test_should_smooth_the_gian_solution_using_sliding_window_mean():
    upstream_output = UpstreamOutput()
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")

    rolled_array_mock.mean.return_value = smooth_gain_mock
    gaintable_mock.gain.rolling.return_value = rolled_array_mock
    upstream_output.gaintable = gaintable_mock

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", False, "./output/path"
    )

    gaintable_mock.gain.rolling.assert_called_once_with(
        frequency=3, center=True
    )
    rolled_array_mock.mean.assert_called_once_with()
    gaintable_mock.assign.assert_called_once_with({"gain": smooth_gain_mock})


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.smooth.plot_gaintable"
)
def test_should_plot_the_smoothed_gain_solution(plot_gaintable_mock):
    upstream_output = UpstreamOutput()
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")

    rolled_array_mock.mean.return_value = smooth_gain_mock
    gaintable_mock.gain.rolling.return_value = rolled_array_mock
    upstream_output.gaintable = gaintable_mock

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", True, "./output/path"
    )

    plot_gaintable_mock.assert_called_once_with(
        smooth_gain_mock,
        "./output/path/smoothed-gaintable",
        figure_title="Smoothed Gain",
        drop_cross_pols=False,
    )
