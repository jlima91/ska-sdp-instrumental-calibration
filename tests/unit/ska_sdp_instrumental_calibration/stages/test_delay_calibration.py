import pytest
from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.delay_calibration import (
    PlotConfig,
    delay_calibration_stage,
)


@pytest.fixture
def plot_config():
    return PlotConfig(plot_table=False, fixed_axis=False)


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "delay_calibration": {
            "oversample": 1,
            "plot_config": {"plot_table": True, "fixed_axis": False},
            "export_gaintable": True,
        },
    }

    assert delay_calibration_stage.__stage__.config == expected_config


def test_delay_calibration_stage_is_not_optional():
    assert delay_calibration_stage.__stage__.is_enabled


@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".calculate_delay"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "delay_calibration.apply_delay_to_gaintable"
)
def test_should_perform_delay_calibration(
    apply_delay_mock,
    calculate_delay_mock,
    create_gaintable_mock,
    apply_gaintable_to_dataset_mock,
    plot_config,
):
    vis_mock = Mock(name="vis")
    initialtable_mock = Mock(name="initialtable")
    gaintable_mock = Mock(name="gaintable")
    gaintable_without_delay_mock = Mock(name="gaintable_without_delay")
    delay_correction_mock = Mock(name="delay_correction")
    delaytable_mock = Mock(name="delaytable")

    create_gaintable_mock.return_value = initialtable_mock
    calculate_delay_mock.return_value = delaytable_mock

    apply_delay_mock.side_effect = [
        gaintable_without_delay_mock,
        delay_correction_mock,
    ]

    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = vis_mock
    upstream_output["gaintable"] = gaintable_mock
    upstream_output["refant"] = "refant"
    oversample = 16

    output = delay_calibration_stage(
        upstream_output,
        _qa_dir_="/output/path",
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=False,
    )

    calculate_delay_mock.assert_called_once_with(gaintable_mock, oversample)
    apply_delay_mock.assert_has_calls(
        [
            call(gaintable_mock, delaytable_mock, inverse=True),
            call(initialtable_mock, delaytable_mock),
        ]
    )

    apply_gaintable_to_dataset_mock.assert_called_once_with(
        vis_mock, delay_correction_mock, inverse=True
    )

    assert output.vis == apply_gaintable_to_dataset_mock.return_value
    assert output.delay == delay_correction_mock
    assert output.gaintable == gaintable_without_delay_mock
    assert output.calibration_tables == ["delay"]


@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".plot_station_delays"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".PlotGaintableFrequency"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".calculate_delay"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "delay_calibration.apply_delay_to_gaintable"
)
def test_should_plot_the_delayed_gaintable_with_proper_suffix(
    apply_delay_mock,
    calculate_delay_mock,
    plot_gaintable_freq_mock,
    plot_station_delays_mock,
    get_plots_path_mock,
    create_gaintable_mock,
    apply_gaintable_to_dataset_mock,
    plot_config,
):
    delaytable_mock = Mock(name="delaytable")
    delayed_gaintable_mock = Mock(name="delayed_gaintable")
    calculate_delay_mock.return_value = delaytable_mock
    apply_delay_mock.return_value = delayed_gaintable_mock

    get_plots_path_mock.side_effect = [
        "/output/path/plots/delay",
        "/output/path/plots/delay_1",
    ]
    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["gaintable"] = Mock(name="gaintable")
    upstream_output["refant"] = 2
    oversample = 16
    plot_gaintable_freq_mock.return_value = plot_gaintable_freq_mock
    plot_gaintable_freq_mock.plot.return_value = ["GAIN_PLOT", "LEAKAGE_PLOT"]
    plot_config.plot_table = True
    plot_config.fixed_axis = True

    delay_calibration_stage(
        upstream_output,
        _qa_dir_="/output/path",
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=False,
    )

    delay_calibration_stage(
        upstream_output,
        _qa_dir_="/output/path",
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=False,
    )

    get_plots_path_mock.assert_has_calls(
        [
            call("/output/path", "ms_prefix/delay"),
            call("/output/path", "ms_prefix/delay_1"),
        ]
    )

    plot_gaintable_freq_mock.assert_has_calls(
        [
            call(
                path_prefix="/output/path/plots/delay",
                refant=2,
            ),
            call.plot(
                delayed_gaintable_mock,
                figure_title="Delay",
                fixed_axis=True,
            ),
            call(
                path_prefix="/output/path/plots/delay_1",
                refant=2,
            ),
            call.plot(
                delayed_gaintable_mock,
                figure_title="Delay",
                fixed_axis=True,
            ),
        ]
    )

    plot_station_delays_mock.assert_has_calls(
        [
            call(
                delaytable_mock,
                "/output/path/plots/delay",
            ),
            call(
                delaytable_mock,
                "/output/path/plots/delay_1",
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".export_clock_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".get_gaintables_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.delay_calibration"
    ".calculate_delay"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "delay_calibration.apply_delay_to_gaintable"
)
def test_should_export_gaintable_with_proper_suffix(
    apply_delay_mock,
    calculate_delay_mock,
    export_gaintable_mock,
    get_gaintables_path_mock,
    delay_mock,
    export_clock_mock,
    create_gaintable_mock,
    apply_gaintable_to_dataset_mock,
    plot_config,
):
    delayed_gaintable_mock = Mock(name="delayed_gaintable")
    apply_delay_mock.return_value = delayed_gaintable_mock

    get_gaintables_path_mock.side_effect = [
        "/output/path/gaintables/delay.gaintable.h5parm",
        "/output/path/gaintables/delay.clock.h5parm",
        "/output/path/gaintables/delay_1.gaintable.h5parm",
        "/output/path/gaintables/delay_1.clock.h5parm",
    ]
    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["gaintable"] = Mock(name="gaintable")
    upstream_output["refant"] = 2
    oversample = 16

    delay_calibration_stage(
        upstream_output,
        _qa_dir_="/output/path",
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=True,
    )

    delay_calibration_stage(
        upstream_output,
        _qa_dir_="/output/path",
        oversample=oversample,
        plot_config=plot_config,
        export_gaintable=True,
    )

    get_gaintables_path_mock.assert_has_calls(
        [
            call("/output/path", "ms_prefix/delay.gaintable.h5parm"),
            call("/output/path", "ms_prefix/delay.clock.h5parm"),
            call("/output/path", "ms_prefix/delay_1.gaintable.h5parm"),
            call("/output/path", "ms_prefix/delay_1.clock.h5parm"),
        ]
    )
    export_gaintable_mock.assert_has_calls(
        [
            call(
                delayed_gaintable_mock,
                "/output/path/gaintables/delay.gaintable.h5parm",
            ),
            call(
                delayed_gaintable_mock,
                "/output/path/gaintables/delay_1.gaintable.h5parm",
            ),
        ]
    )

    delay_mock.assert_has_calls(
        [call(export_gaintable_mock), call(export_gaintable_mock)]
    )
