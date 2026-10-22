# flake8: noqa:E501
import pytest
from mock import Mock

from ska_sdp_instrumental_calibration.xarray_processors.gain_smoothing import (
    sliding_window_smooth,
)


def test_sliding_window_smooth_with_mean():
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")
    chunked_smooth_gain_mock = Mock(name="chunked_smoothened_array")
    smooth_gain_mock.chunk.return_value = chunked_smooth_gain_mock

    rolled_array_mock.mean.return_value = smooth_gain_mock
    gaintable_mock.CALPARAM_GAIN.rolling.return_value = rolled_array_mock
    gaintable_mock.CALPARAM_GAIN.chunksizes = "chunksizes"

    sliding_window_smooth(gaintable_mock, 3, "mean")

    gaintable_mock.CALPARAM_GAIN.rolling.assert_called_once_with(
        frequency=3, center=True
    )
    rolled_array_mock.mean.assert_called_once_with()
    gaintable_mock.assign.assert_called_once_with(
        {"CALPARAM_GAIN": chunked_smooth_gain_mock}
    )
    smooth_gain_mock.chunk.assert_called_once_with("chunksizes")


def test_sliding_window_smooth_with_median():
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")
    chunked_smooth_gain_mock = Mock(name="chunked_smoothened_array")
    smooth_gain_mock.chunk.return_value = chunked_smooth_gain_mock

    rolled_array_mock.median.return_value = smooth_gain_mock
    gaintable_mock.CALPARAM_GAIN.rolling.return_value = rolled_array_mock
    gaintable_mock.CALPARAM_GAIN.chunksizes = "chunksizes"

    sliding_window_smooth(gaintable_mock, 3, "median")

    gaintable_mock.CALPARAM_GAIN.rolling.assert_called_once_with(
        frequency=3, center=True
    )
    rolled_array_mock.median.assert_called_once_with()
    gaintable_mock.assign.assert_called_once_with(
        {"CALPARAM_GAIN": chunked_smooth_gain_mock}
    )
    smooth_gain_mock.chunk.assert_called_once_with("chunksizes")


def test_sliding_window_smooth_with_invalid_mode():
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")

    gaintable_mock.CALPARAM_GAIN.rolling.return_value = rolled_array_mock

    with pytest.raises(ValueError) as error:
        sliding_window_smooth(gaintable_mock, 3, "invalid")
        assert error.msg == "Unsupported sliding window smooth mode invalid"
