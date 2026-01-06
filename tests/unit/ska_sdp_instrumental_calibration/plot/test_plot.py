from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

from ska_sdp_instrumental_calibration.plot.plot import plot_curve_fit


@pytest.fixture
def gaintable_mock():
    gain_isel_mock = Mock(name="gain_isel_mock")

    gain_mock = Mock(name="gain_mock")
    gain_mock.isel.return_value = gain_isel_mock

    gaintable_mock = Mock(name="gaintable_mock")
    gaintable_mock.gain = gain_mock
    gaintable_mock.stack.return_value = gaintable_mock
    gaintable_mock.assign_coords.return_value = gaintable_mock

    gaintable_mock.pol.data = [("X", "X"), ("Y", "Y"), ("X", "Y"), ("Y", "X")]
    gaintable_mock.pol.values = ["J_XX", "J_YY", "J_XY", "J_YX"]

    stations_mock = Mock(name="stations_mock")
    stations_mock.values = np.array(["A1"])
    stations_mock.id = np.array([0])
    stations_mock.size = 1

    gaintable_mock.configuration.names = stations_mock

    frequency_mock = MagicMock(name="frequency_mock")
    frequency_mock.__len__.return_value = 4
    frequency_mhz_mock = MagicMock(name="frequency_mhz_mock")
    frequency_mhz_mock.__len__.return_value = 4
    frequency_mock.__truediv__.return_value = frequency_mhz_mock
    gaintable_mock.frequency = frequency_mock

    return gaintable_mock


def make_fit(name):
    slice_mock = Mock(name=f"{name}_slice")

    isel_mock = MagicMock(name=f"{name}_isel")
    isel_mock.__getitem__.return_value = slice_mock

    stack_mock = Mock(name=f"{name}_stack")
    stack_mock.isel.return_value = isel_mock

    fit_mock = Mock(name=f"{name}_fit")
    fit_mock.stack.return_value = stack_mock

    return fit_mock, slice_mock


@patch("ska_sdp_instrumental_calibration.plot.plot.plt.close")
@patch("ska_sdp_instrumental_calibration.plot.plot.plt.get_cmap")
@patch("ska_sdp_instrumental_calibration.plot.plot.plt.figure")
@patch("ska_sdp_instrumental_calibration.plot.plot.normalize_data")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.angle")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.absolute")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.split")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.arange")
@patch(
    "ska_sdp_instrumental_calibration.plot.plot.np.rad2deg",
    new=lambda x: x,
)
def test_plot_curve_fit_amp_phase_normalized(
    arange_mock,
    split_mock,
    absolute_mock,
    angle_mock,
    normalize_data_mock,
    figure_mock,
    get_cmap_mock,
    close_mock,
    gaintable_mock,
):
    amp_fit_mock, amp_fit_slice = make_fit("amp")
    phase_fit_mock, phase_fit_slice = make_fit("phase")

    fits = {
        "amp_fit": amp_fit_mock,
        "phase_fit": phase_fit_mock,
        "real_fit": Mock(name="real_fit"),
        "imag_fit": Mock(name="imag_fit"),
    }

    amp_data_mock = MagicMock(name="amp_data_mock")
    amp_slice_mock = Mock(name="amp_data_slice_mock")
    amp_data_mock.__getitem__.return_value = amp_slice_mock

    phase_data_mock = MagicMock(name="phase_data_mock")
    phase_data_mock.__getitem__.return_value = Mock(
        name="phase_data_slice_mock"
    )

    normalized_amp_mock = Mock(name="normalized_amp_mock")
    normalize_data_mock.return_value = normalized_amp_mock

    absolute_mock.return_value = amp_data_mock
    angle_mock.return_value = phase_data_mock

    channel_mock = Mock(name="channel_mock")
    arange_mock.return_value = channel_mock

    fig_mock = Mock(name="fig_mock")
    subfig_mock = Mock(name="subfig_mock")

    top_ax_mock = Mock(name="top_ax_mock")
    bottom_ax_mock = Mock(name="bottom_ax_mock")

    split_mock.return_value = [gaintable_mock.configuration.names]

    subfigures_mock = Mock(name="subfigures_mock")
    subfigures_mock.reshape.return_value = [subfig_mock]
    fig_mock.subfigures.return_value = subfigures_mock

    subfig_mock.subplots.return_value = np.array(
        [
            [top_ax_mock, Mock(name="ax_unused_1")],
            [bottom_ax_mock, Mock(name="ax_unused_2")],
        ]
    )

    figure_mock.return_value = fig_mock

    cmap_mock = Mock(name="cmap_mock")
    color_0_mock = Mock(name="color_0_mock")
    cmap_mock.side_effect = lambda i: color_0_mock
    get_cmap_mock.return_value = cmap_mock

    plot_curve_fit(
        gaintable_mock,
        fits,
        soltype="amp-phase",
        normalize_gains=True,
        path_prefix="/tmp/test",
    ).compute()

    gain = gaintable_mock.gain.isel.return_value
    antenna_gain = gain.isel.return_value

    gaintable_mock.gain.isel.assert_called_once_with(time=0)
    absolute_mock.assert_called_once_with(antenna_gain)
    angle_mock.assert_called_once_with(antenna_gain, deg=True)

    normalize_data_mock.assert_has_calls(
        [call(amp_data_mock.__getitem__.return_value.values)]
    )
    bottom_ax_mock.scatter.assert_has_calls(
        [
            call(
                channel_mock,
                normalized_amp_mock,
                color=color_0_mock,
                label="J_XX",
                alpha=0.4,
                s=15,
            )
        ]
    )

    fits["amp_fit"].stack.assert_called_once_with(
        pol=("receptor1", "receptor2")
    )
    fits["phase_fit"].stack.assert_called_once_with(
        pol=("receptor1", "receptor2")
    )

    top_ax_mock.plot.assert_has_calls(
        [
            call(
                channel_mock,
                phase_fit_slice,
                color=color_0_mock,
                label="J_XX",
                lw=2,
            )
        ]
    )

    bottom_ax_mock.plot.assert_has_calls(
        [
            call(
                channel_mock,
                amp_fit_slice,
                color=color_0_mock,
                label="J_XX",
                lw=2,
            )
        ]
    )

    close_mock.assert_called_once()


@patch("ska_sdp_instrumental_calibration.plot.plot.plt.close")
@patch("ska_sdp_instrumental_calibration.plot.plot.plt.get_cmap")
@patch("ska_sdp_instrumental_calibration.plot.plot.plt.figure")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.imag")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.real")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.split")
@patch("ska_sdp_instrumental_calibration.plot.plot.np.arange")
def test_plot_curve_fit_real_imag_scatter_and_plot_args_exact(
    arange_mock,
    split_mock,
    real_mock,
    imag_mock,
    figure_mock,
    get_cmap_mock,
    close_mock,
    gaintable_mock,
):
    real_fit_mock, real_fit_slice = make_fit("real")
    imag_fit_mock, imag_fit_slice = make_fit("imag")

    fits = {
        "real_fit": real_fit_mock,
        "imag_fit": imag_fit_mock,
        "amp_fit": Mock(name="amp_fit"),
        "phase_fit": Mock(name="phase_fit"),
    }

    real_data_mock = MagicMock(name="real_data_mock")
    real_data_mock.__getitem__.return_value = Mock(name="real_data_slice")

    imag_data_mock = MagicMock(name="imag_data_mock")
    imag_data_mock.__getitem__.return_value = Mock(name="imag_data_slice")

    real_mock.return_value = real_data_mock
    imag_mock.return_value = imag_data_mock

    channel_mock = Mock(name="channel_mock")
    arange_mock.return_value = channel_mock

    fig_mock = Mock(name="fig_mock")
    subfig_mock = Mock(name="subfig_mock")

    top_ax_mock = Mock(name="top_ax_mock")
    bottom_ax_mock = Mock(name="bottom_ax_mock")

    split_mock.return_value = [gaintable_mock.configuration.names]

    subfigures_mock = Mock(name="subfigures_mock")
    subfigures_mock.reshape.return_value = [subfig_mock]
    fig_mock.subfigures.return_value = subfigures_mock

    subfig_mock.subplots.return_value = np.array(
        [
            [top_ax_mock, Mock(name="ax_unused_1")],
            [bottom_ax_mock, Mock(name="ax_unused_2")],
        ]
    )

    figure_mock.return_value = fig_mock

    cmap_mock = Mock(name="cmap_mock")
    color_0_mock = Mock(name="color_0_mock")
    cmap_mock.side_effect = lambda i: color_0_mock
    get_cmap_mock.return_value = cmap_mock

    plot_curve_fit(
        gaintable_mock,
        fits,
        soltype="real-imag",
        path_prefix="/tmp/test",
    ).compute()

    gain = gaintable_mock.gain.isel.return_value
    antenna_gain = gain.isel.return_value

    gaintable_mock.gain.isel.assert_called_once_with(time=0)

    real_mock.assert_called_once_with(antenna_gain)
    imag_mock.assert_called_once_with(antenna_gain)

    real_fit_mock.stack.assert_called_once_with(pol=("receptor1", "receptor2"))
    imag_fit_mock.stack.assert_called_once_with(pol=("receptor1", "receptor2"))

    top_ax_mock.scatter.assert_has_calls(
        [
            call(
                channel_mock,
                imag_data_mock.__getitem__.return_value,
                color=color_0_mock,
                label="J_XX",
                alpha=0.4,
                s=15,
            )
        ]
    )

    bottom_ax_mock.scatter.assert_has_calls(
        [
            call(
                channel_mock,
                real_data_mock.__getitem__.return_value.values,
                color=color_0_mock,
                label="J_XX",
                alpha=0.4,
                s=15,
            )
        ]
    )

    top_ax_mock.plot.assert_has_calls(
        [
            call(
                channel_mock,
                imag_fit_slice,
                color=color_0_mock,
                label="J_XX",
                lw=2,
            )
        ]
    )

    bottom_ax_mock.plot.assert_has_calls(
        [
            call(
                channel_mock,
                real_fit_slice,
                color=color_0_mock,
                label="J_XX",
                lw=2,
            )
        ]
    )

    close_mock.assert_called_once()
