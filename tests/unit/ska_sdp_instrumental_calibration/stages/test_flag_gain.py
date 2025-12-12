from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import flag_gain_stage


def test_flag_gain_stage_is_optional():
    assert flag_gain_stage.is_optional


@patch("ska_sdp_instrumental_calibration.stages.flag_gain.plot_flag_gain")
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.plot_curve_fit")
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.flag_on_gains")
def test_should_perform_flagging_on_gains(
    flag_on_gains_mock, plot_curve_mock, plot_flag_mock
):
    upstream_output = UpstreamOutput()
    initialtable = Mock(name="initial_gaintable")
    upstream_output["gaintable"] = initialtable
    soltype = "amplitude"
    mode = "smooth"
    order = 3
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    normalize_gains = False
    apply_flag = True
    skip_cross_pol = False
    export_gaintable = False
    plot_config = {"curve_fit_plot": False, "gain_flag_plot": False}

    gaintable_mock = Mock(name="gaintable")
    amp_fit_mock = Mock(name="Amp fit")
    phase_fit_mock = Mock(name="Phase fit")
    flag_on_gains_mock.return_value = (
        gaintable_mock,
        amp_fit_mock,
        phase_fit_mock,
    )

    actual = flag_gain_stage.stage_definition(
        upstream_output,
        soltype,
        mode,
        order,
        skip_cross_pol,
        export_gaintable,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        apply_flag,
        plot_config,
        _output_dir_="/output/path",
    )

    flag_on_gains_mock.assert_called_once_with(
        initialtable,
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        skip_cross_pol,
        apply_flag,
    )

    assert actual.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.stages.flag_gain.dask.delayed",
    side_effect=lambda x: x,
)
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.get_gaintables_path")
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.plot_flag_gain")
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.plot_curve_fit")
@patch(
    "ska_sdp_instrumental_calibration.stages.flag_gain"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.flag_on_gains")
def test_should_export_gaintable_with_proper_suffix(
    flag_on_gains_mock,
    export_gaintable_mock,
    plot_curve_mock,
    plot_flag_mock,
    get_gaintables_path_mock,
    delayed_mock,
):
    get_gaintables_path_mock.side_effect = [
        "/output/path/gaintables/gain_flag.gaintable.h5parm",
        "/output/path/gaintables/gain_flag_1.gaintable.h5parm",
    ]

    upstream_output = UpstreamOutput()
    initialtable = Mock(name="initial_gaintable")
    upstream_output["gaintable"] = initialtable
    soltype = "amplitude"
    mode = "smooth"
    order = 3
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    normalize_gains = False
    apply_flag = True
    skip_cross_pol = False
    export_gaintable = True
    plot_config = {"curve_fit_plot": False, "gain_flag_plot": False}

    gaintable_mock = Mock(name="gaintable")
    amp_fit_mock = Mock(name="Amp fit")
    phase_fit_mock = Mock(name="Phase fit")
    flag_on_gains_mock.return_value = (
        gaintable_mock,
        amp_fit_mock,
        phase_fit_mock,
    )

    flag_gain_stage.stage_definition(
        upstream_output,
        soltype,
        mode,
        order,
        skip_cross_pol,
        export_gaintable,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        apply_flag,
        plot_config,
        _output_dir_="/output/path",
    )

    flag_gain_stage.stage_definition(
        upstream_output,
        soltype,
        mode,
        order,
        skip_cross_pol,
        export_gaintable,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        apply_flag,
        plot_config,
        _output_dir_="/output/path",
    )

    get_gaintables_path_mock.assert_has_calls(
        [
            call("/output/path", "gain_flag.gaintable.h5parm"),
            call("/output/path", "gain_flag_1.gaintable.h5parm"),
        ]
    )
    export_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "/output/path/gaintables/gain_flag.gaintable.h5parm",
            ),
            call(
                gaintable_mock,
                "/output/path/gaintables/gain_flag_1.gaintable.h5parm",
            ),
        ]
    )

    delayed_mock.assert_has_calls(
        [call(export_gaintable_mock), call(export_gaintable_mock)]
    )


@patch(
    "ska_sdp_instrumental_calibration.stages.flag_gain.dask.delayed",
    side_effect=lambda x: x,
)
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.get_plots_path")
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.plot_flag_gain")
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.plot_curve_fit")
@patch(
    "ska_sdp_instrumental_calibration.stages.flag_gain"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.stages.flag_gain.flag_on_gains")
def test_should_plot_flag_on_gain(
    flag_on_gains_mock,
    export_gaintable_mock,
    plot_curve_mock,
    plot_flag_mock,
    get_plots_path_mock,
    delayed_mock,
):
    get_plots_path_mock.side_effect = [
        "/output/path/plots/gain_flagging",
        "/output/path/plots/curve_fit_gain",
    ]
    upstream_output = UpstreamOutput()
    initialtable = Mock(name="initial_gaintable")
    upstream_output["gaintable"] = initialtable
    soltype = "amplitude"
    mode = "smooth"
    order = 3
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    normalize_gains = False
    apply_flag = True
    skip_cross_pol = False
    export_gaintable = False
    plot_config = {"curve_fit_plot": True, "gain_flag_plot": True}

    gaintable_mock = Mock(name="gaintable")
    amp_fit_mock = Mock(name="Amp fit")
    phase_fit_mock = Mock(name="Phase fit")
    flag_on_gains_mock.return_value = (
        gaintable_mock,
        amp_fit_mock,
        phase_fit_mock,
    )

    flag_gain_stage.stage_definition(
        upstream_output,
        soltype,
        mode,
        order,
        skip_cross_pol,
        export_gaintable,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        apply_flag,
        plot_config,
        _output_dir_="/output/path",
    )

    get_plots_path_mock.assert_has_calls(
        [
            call("/output/path", "gain_flagging"),
            call("/output/path", "curve_fit_gain"),
        ]
    )

    plot_flag_mock.assert_called_once_with(
        gaintable_mock,
        "/output/path/plots/gain_flagging",
        figure_title="Gain Flagging",
    )

    plot_curve_mock.assert_called_once_with(
        gaintable_mock,
        amp_fit_mock,
        phase_fit_mock,
        "/output/path/plots/curve_fit_gain",
        normalize_gains,
        figure_title="Curve fit of Gain Flagging",
    )
