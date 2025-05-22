from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    smooth_gain_solution_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
def test_should_smooth_the_gain_solution(sliding_window_smooth_mock):
    upstream_output = UpstreamOutput()

    gaintable_mock = Mock(name="gaintable")
    upstream_output.gaintable = gaintable_mock

    sliding_window_smooth_mock.return_value = gaintable_mock

    plot_config = {
        "plot_table": False,
        "plot_path_prefix": "./some/path",
        "plot_title": "plot title",
    }

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "median", plot_config, False, "./output/path"
    )

    sliding_window_smooth_mock.assert_called_once_with(
        upstream_output.gaintable, 3, "median"
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
def test_should_smooth_the_gain_solution_using_sliding_window_mean(
    sliding_window_smooth_mock,
):
    upstream_output = UpstreamOutput()

    gaintable_mock = Mock(name="gaintable")
    upstream_output.gaintable = gaintable_mock

    sliding_window_smooth_mock.return_value = gaintable_mock

    plot_config = {
        "plot_table": False,
        "plot_path_prefix": "./some/path",
        "plot_title": "plot title",
    }

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", plot_config, False, "./output/path"
    )

    sliding_window_smooth_mock.assert_called_once_with(
        upstream_output.gaintable, 3, "mean"
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.plot_gaintable"
)
def test_should_plot_the_smoothed_gain_solution(
    plot_gaintable_mock, sliding_window_smooth_mock
):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")

    sliding_window_smooth_mock.return_value = gaintable_mock

    upstream_output.gaintable = gaintable_mock

    plot_config = {
        "plot_table": True,
        "plot_path_prefix": "some/path",
        "plot_title": "plot title",
    }

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", plot_config, False, "./output/path"
    )

    plot_gaintable_mock.assert_called_once_with(
        gaintable_mock,
        "./output/path/some/path",
        figure_title="plot title",
        drop_cross_pols=False,
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".smooth_gain_solution.plot_gaintable"
)
def test_should_plot_smoothed_gain_solution_with_suffix(
    plot_gaintable_mock, sliding_window_smooth_mock
):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")

    upstream_output.gaintable = gaintable_mock
    sliding_window_smooth_mock.return_value = gaintable_mock

    plot_config = {
        "plot_table": True,
        "plot_path_prefix": "some/path",
        "plot_title": "plot title",
    }

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", plot_config, False, "./output/path"
    )

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", plot_config, False, "./output/path"
    )

    plot_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "./output/path/some/path",
                figure_title="plot title",
                drop_cross_pols=False,
            ),
            call(
                gaintable_mock,
                "./output/path/some/path_1",
                figure_title="plot title",
                drop_cross_pols=False,
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".smooth_gain_solution.export_gaintable_to_h5parm"
)
def test_should_export_smoothed_gain_solution_with_suffix(
    export_gaintable_mock, sliding_window_smooth_mock, dask_delayed_mock
):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")

    upstream_output.gaintable = gaintable_mock
    sliding_window_smooth_mock.return_value = gaintable_mock

    plot_config = {
        "plot_table": False,
        "plot_path_prefix": "some/path",
        "plot_title": "plot title",
    }

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", plot_config, True, "./output/path"
    )

    smooth_gain_solution_stage.stage_definition(
        upstream_output, 3, "mean", plot_config, True, "./output/path"
    )

    export_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "./output/path/smooth_gain.gaintable.h5parm",
            ),
            call(
                gaintable_mock,
                "./output/path/smooth_gain_1.gaintable.h5parm",
            ),
        ]
    )
