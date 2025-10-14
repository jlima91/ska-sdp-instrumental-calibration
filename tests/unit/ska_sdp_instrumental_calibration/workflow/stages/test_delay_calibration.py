from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    delay_calibration_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".calculate_delay"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_perform_delay_calibration(
    apply_delay_mock, calculate_delay_mock
):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    oversample = 16
    plot_config = {"plot_table": False, "fixed_axis": False}

    actual_output = delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    calculate_delay_mock.assert_called_once_with(gaintable_mock, oversample)
    apply_delay_mock.assert_called_once_with(
        gaintable_mock, calculate_delay_mock.return_value
    )

    assert actual_output.gaintable == apply_delay_mock.return_value


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".plot_station_delays"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".calculate_delay"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_plot_the_delayed_gaintable_with_proper_suffix(
    apply_delay_mock,
    calculate_delay_mock,
    plot_gaintable_mock,
    plot_station_delays_mock,
    get_plots_path_mock,
):
    get_plots_path_mock.side_effect = [
        "/output/path/plots/delay",
        "/output/path/plots/delay_1",
    ]
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    oversample = 16
    plot_config = {"plot_table": True, "fixed_axis": True}

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    get_plots_path_mock.assert_has_calls(
        [
            call("/output/path", "delay"),
            call("/output/path", "delay_1"),
        ]
    )

    plot_gaintable_mock.assert_has_calls(
        [
            call(
                apply_delay_mock.return_value,
                "/output/path/plots/delay",
                figure_title="Delay",
                fixed_axis=True,
            ),
            call(
                apply_delay_mock.return_value,
                "/output/path/plots/delay_1",
                figure_title="Delay",
                fixed_axis=True,
            ),
        ]
    )

    plot_station_delays_mock.assert_has_calls(
        [
            call(
                calculate_delay_mock.return_value,
                "/output/path/plots/delay",
                show_station_label=False,
            ),
            call(
                calculate_delay_mock.return_value,
                "/output/path/plots/delay_1",
                show_station_label=False,
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".get_gaintables_path"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".calculate_delay"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_export_gaintable_with_proper_suffix(
    apply_delay_mock,
    calculate_delay_mock,
    export_gaintable_mock,
    get_gaintables_path_mock,
    delay_mock,
):
    get_gaintables_path_mock.side_effect = [
        "/output/path/gaintables/delay.gaintable.h5parm",
        "/output/path/gaintables/delay.clock.h5parm",
        "/output/path/gaintables/delay_1.gaintable.h5parm",
        "/output/path/gaintables/delay_1.clock.h5parm",
    ]
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    oversample = 16
    plot_config = {"plot_table": False, "fixed_axis": True}

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    get_gaintables_path_mock.assert_has_calls(
        [
            call("/output/path", "delay.gaintable.h5parm"),
            call("/output/path", "delay.clock.h5parm"),
            call("/output/path", "delay_1.gaintable.h5parm"),
            call("/output/path", "delay_1.clock.h5parm"),
        ]
    )
    export_gaintable_mock.assert_has_calls(
        [
            call(
                apply_delay_mock.return_value,
                "/output/path/gaintables/delay.gaintable.h5parm",
            ),
            call(
                apply_delay_mock.return_value,
                "/output/path/gaintables/delay_1.gaintable.h5parm",
            ),
        ]
    )

    delay_mock.assert_has_calls(
        [call(export_gaintable_mock), call(export_gaintable_mock)]
    )
