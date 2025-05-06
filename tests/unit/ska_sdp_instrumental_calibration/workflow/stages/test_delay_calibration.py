from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    delay_calibration_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_perform_delay_calibration(apply_delay_mock, dask_delayed_mock):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    delayed_gaintable_mock = Mock(name="delayed_gaintable")
    oversample = 16
    plot_config = {"plot_table": False, "fixed_axis": False}

    apply_delay_mock.return_value = delayed_gaintable_mock

    actual_output = delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        _output_dir_="/output/path",
    )

    apply_delay_mock.assert_called_once_with(gaintable_mock, oversample)

    dask_delayed_mock.assert_called_once_with(apply_delay_mock)

    assert actual_output.gaintable == delayed_gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.delay_calibration"
    ".apply_delay"
)
def test_should_plot_the_delayed_gaintable(
    apply_delay_mock, plot_gaintable_mock, dask_delayed_mock
):
    upstream_output = UpstreamOutput()
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    delayed_gaintable_mock = Mock(name="delayed_gaintable")
    oversample = 16
    plot_config = {"plot_table": True, "fixed_axis": True}

    apply_delay_mock.return_value = delayed_gaintable_mock

    delay_calibration_stage.stage_definition(
        upstream_output,
        oversample=oversample,
        plot_config=plot_config,
        _output_dir_="/output/path",
    )

    plot_gaintable_mock.assert_called_once_with(
        delayed_gaintable_mock,
        "/output/path/delay",
        figure_title="Delay",
        fixed_axis=True,
    )
