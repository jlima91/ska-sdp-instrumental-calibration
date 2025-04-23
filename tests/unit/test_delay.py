import numpy as np
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.processing_tasks.delay import (
    apply_delay,
    calculate_gain_rot,
    coarse_delay,
    update_delay,
)


def test_should_calculate_coarse_delay():
    oversample = 8
    nstations = 20
    nchan = 4
    frequency = xr.DataArray(np.linspace(100e6, 150e6, nchan))
    gains_per_station = np.ones(nchan) + 1j * np.ones(nchan)
    gains = np.stack([gains_per_station] * nstations)

    expected = np.zeros(nstations)
    actual = coarse_delay(frequency, gains, oversample)

    assert np.allclose(expected, actual, atol=1e-11)


@patch("ska_sdp_instrumental_calibration.processing_tasks.delay.np")
def test_coarse_delay(np_mock):
    freq_mock = MagicMock()
    gain_mock = MagicMock(name="gain_mock")
    gain_mock.shape = (20, 64)
    oversample = 16
    zeros_mock = MagicMock(name="zeros_mock")
    fft_value_mock = MagicMock()
    fft_shift_value_mock = MagicMock()
    abs_value_mock = MagicMock()
    arange_value_mock = MagicMock()

    np_mock.zeros.return_value = zeros_mock
    np_mock.fft.fft.return_value = fft_value_mock
    np_mock.fft.fftshift.return_value = fft_shift_value_mock
    np_mock.arange.return_value = arange_value_mock
    np_mock.abs.return_value = abs_value_mock

    coarse_delay(freq_mock, gain_mock, oversample)

    np_mock.zeros.assert_called_once_with((20, 1024), "complex")
    np_mock.fft.fft.assert_called_once_with(zeros_mock, axis=1)
    np_mock.fft.fftshift.assert_called_once_with(fft_value_mock, axes=(1,))
    np_mock.arange.assert_called_once_with(1024)
    np_mock.abs.assert_called_once_with(fft_shift_value_mock)


def test_calculate_gain_rotation():
    frequency = np.linspace(100e6, 200e6, 4).reshape(1, -1)
    f = np.linspace(4, 10, 4) + 1j * np.linspace(3, 9, 4)
    gains = np.stack([f] * 2)
    delay = np.ones(2)
    offset = np.ones(2)

    expected = [
        [
            3.99999993 + 3.00000009j,
            1.33012709 - 7.69615241j,
            -10.06217822 + 3.42820208j,
            10.00000017 + 8.99999981j,
        ],
        [
            3.99999993 + 3.00000009j,
            1.33012709 - 7.69615241j,
            -10.06217822 + 3.42820208j,
            10.00000017 + 8.99999981j,
        ],
    ]
    actual = calculate_gain_rot(gains, delay, offset, frequency)

    assert np.allclose(actual, expected)


@patch("ska_sdp_instrumental_calibration.processing_tasks.delay.np")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.delay"
    ".calculate_gain_rot"
)
def test_update_delay(cal_gain_rot_mock, np_mock):

    freq_mock = MagicMock(name="freq_mock")
    gains_mock = MagicMock(name="gains_mock")
    weights_mock = MagicMock(name="weights_mock")
    gains_rot_mock = MagicMock(name="gains_rot_mock")
    cycles_mock = MagicMock(name="cycles_mock")
    sum_result_mock = MagicMock(name="sum_result_mock")
    offset_mock = MagicMock(name="offset_mock")
    delay_mock = MagicMock(name="delay_mock")
    gaintable = MagicMock(name="gaintable")

    np_mock.unwrap.return_value = cycles_mock
    np_mock.angle.return_value = cycles_mock
    np_mock.sum.return_value = sum_result_mock
    gaintable.frequency.data.reshape.return_value = freq_mock
    gaintable.gain.data.__getitem__.return_value = gains_mock
    gaintable.weight.data.__getitem__.return_value = weights_mock
    cal_gain_rot_mock.return_value = gains_rot_mock

    update_delay(gaintable, offset_mock, delay_mock, 0)

    cal_gain_rot_mock.assert_called_once_with(
        gains_mock, delay_mock, offset_mock, freq_mock
    )


@patch("ska_sdp_instrumental_calibration.processing_tasks.delay.np")
@patch("ska_sdp_instrumental_calibration.processing_tasks.delay.update_delay")
@patch("ska_sdp_instrumental_calibration.processing_tasks.delay.coarse_delay")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.delay"
    ".calculate_gain_rot"
)
def test_apply_delay(
    cal_gain_rot_mock, coarse_delay_mock, update_delay_mock, np_mock
):
    gains_mock = MagicMock(name="gains_mock")
    weights_mock = MagicMock(name="weights_mock")
    cycles_mock = MagicMock(name="cycles_mock")
    sum_result_mock = MagicMock(name="sum_result_mock")
    xgain_mock = MagicMock(name="xgain_mock")
    xoffset_mock = MagicMock(name="xoffset_mock")
    xdelay_coarse_mock = MagicMock(name="xdelay_coarse_mock")
    reshape_mock = MagicMock(name="reshape_mock")
    gaintable = MagicMock(name="gaintable")
    np_zeros_mock = MagicMock(name="np_zeros_mock")

    np_mock.unwrap.return_value = cycles_mock
    np_mock.angle.return_value = cycles_mock
    np_mock.sum.return_value = sum_result_mock
    update_delay_mock.return_value = [xgain_mock, xoffset_mock]
    coarse_delay_mock.return_value = xdelay_coarse_mock
    gaintable.frequency.data.__getitem__.side_effect = [xgain_mock, xgain_mock]
    gaintable.frequency.data.reshape.return_value = reshape_mock
    gaintable.gain.data.__getitem__.return_value = gains_mock
    gaintable.weight.data.__getitem__.return_value = weights_mock
    np_mock.zeros.return_value = np_zeros_mock

    apply_delay(gaintable, oversample=16)

    coarse_delay_mock.assert_called_with(gaintable.frequency, gains_mock, 16)
    update_delay_mock.assert_called_with(
        gaintable, np_zeros_mock, coarse_delay_mock.return_value, pol=1
    )
    cal_gain_rot_mock.assert_called_with(
        gains_mock, xgain_mock, xoffset_mock, reshape_mock
    )
