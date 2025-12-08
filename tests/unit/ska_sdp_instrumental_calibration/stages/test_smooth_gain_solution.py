from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import smooth_gain_solution_stage


@patch(
    "ska_sdp_instrumental_calibration.stages."
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
    "ska_sdp_instrumental_calibration.stages."
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
    "ska_sdp_instrumental_calibration.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "smooth_gain_solution.get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "smooth_gain_solution.PlotGaintableFrequency"
)
def test_should_plot_the_smoothed_gain_solution(
    plot_gaintable_freq_mock, get_plots_path_mock, sliding_window_smooth_mock
):
    get_plots_path_mock.return_value = "./output/path/plots/some/path"
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    plot_gaintable_freq_mock.return_value = plot_gaintable_freq_mock

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

    get_plots_path_mock.assert_called_once_with("./output/path", "some/path")
    plot_gaintable_freq_mock.assert_called_once_with(
        path_prefix="./output/path/plots/some/path",
    )
    plot_gaintable_freq_mock.plot.assert_called_once_with(
        gaintable_mock,
        figure_title="plot title",
    )


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
@patch(
    "ska_sdp_instrumental_calibration.stages"
    ".smooth_gain_solution.get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages"
    ".smooth_gain_solution.PlotGaintableFrequency"
)
def test_should_plot_smoothed_gain_solution_with_suffix(
    plot_gaintable_freq_mock, get_plots_path_mock, sliding_window_smooth_mock
):
    get_plots_path_mock.side_effect = [
        "./output/path/plots/some/path",
        "./output/path/plots/some/path_1",
    ]
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    plot_gaintable_freq_mock.return_value = plot_gaintable_freq_mock

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

    get_plots_path_mock.assert_has_calls(
        [
            call("./output/path", "some/path"),
            call("./output/path", "some/path_1"),
        ]
    )

    plot_gaintable_freq_mock.assert_has_calls(
        [
            call(
                path_prefix="./output/path/plots/some/path",
            ),
            call.plot(gaintable_mock, figure_title="plot title"),
            call(
                path_prefix="./output/path/plots/some/path_1",
            ),
            call.plot(gaintable_mock, figure_title="plot title"),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "smooth_gain_solution.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "smooth_gain_solution.sliding_window_smooth"
)
@patch(
    "ska_sdp_instrumental_calibration.stages"
    ".smooth_gain_solution.get_gaintables_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages"
    ".smooth_gain_solution.export_gaintable_to_h5parm"
)
def test_should_export_smoothed_gain_solution_with_suffix(
    export_gaintable_mock,
    get_gaintables_path_mock,
    sliding_window_smooth_mock,
    dask_delayed_mock,
):
    get_gaintables_path_mock.side_effect = [
        "./output/path/gaintables/smooth_gain.gaintable.h5parm",
        "./output/path/gaintables/smooth_gain_1.gaintable.h5parm",
    ]
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

    get_gaintables_path_mock.assert_has_calls(
        [
            call("./output/path", "smooth_gain.gaintable.h5parm"),
            call("./output/path", "smooth_gain_1.gaintable.h5parm"),
        ]
    )

    export_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "./output/path/gaintables/smooth_gain.gaintable.h5parm",
            ),
            call(
                gaintable_mock,
                "./output/path/gaintables/smooth_gain_1.gaintable.h5parm",
            ),
        ]
    )
