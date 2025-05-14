# flake8: noqa:E501
from mock import Mock

from ska_sdp_instrumental_calibration.processing_tasks.sliding_window_smooth import (
    sliding_window_smooth,
)


def test_sliding_window_smooth_with_mean():
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")
    chunked_smooth_gain_mock = Mock(name="chunked_smoothened_array")
    smooth_gain_mock.chunk.return_value = chunked_smooth_gain_mock

    rolled_array_mock.mean.return_value = smooth_gain_mock
    gaintable_mock.gain.rolling.return_value = rolled_array_mock
    gaintable_mock.gain.chunksizes = "chunksizes"

    sliding_window_smooth(gaintable_mock, 3, "mean")

    gaintable_mock.gain.rolling.assert_called_once_with(
        frequency=3, center=True
    )
    rolled_array_mock.mean.assert_called_once_with()
    gaintable_mock.assign.assert_called_once_with(
        {"gain": chunked_smooth_gain_mock}
    )
    smooth_gain_mock.chunk.assert_called_once_with("chunksizes")


def test_sliding_window_smooth_with_median():
    rolled_array_mock = Mock(name="rolled array")
    gaintable_mock = Mock(name="gaintable")
    smooth_gain_mock = Mock(name="smoothened_array")
    chunked_smooth_gain_mock = Mock(name="chunked_smoothened_array")
    smooth_gain_mock.chunk.return_value = chunked_smooth_gain_mock

    rolled_array_mock.median.return_value = smooth_gain_mock
    gaintable_mock.gain.rolling.return_value = rolled_array_mock
    gaintable_mock.gain.chunksizes = "chunksizes"

    sliding_window_smooth(gaintable_mock, 3, "median")

    gaintable_mock.gain.rolling.assert_called_once_with(
        frequency=3, center=True
    )
    rolled_array_mock.median.assert_called_once_with()
    gaintable_mock.assign.assert_called_once_with(
        {"gain": chunked_smooth_gain_mock}
    )
    smooth_gain_mock.chunk.assert_called_once_with("chunksizes")
