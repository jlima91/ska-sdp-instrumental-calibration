import numpy as np
from mock import patch

from ska_sdp_instrumental_calibration.workflow.plot_gaintable import (
    PlotGaintableFrequency,
    PlotGaintableTime,
    safe,
)


@patch("ska_sdp_instrumental_calibration.workflow.plot_gaintable.logger")
@patch("ska_sdp_instrumental_calibration.workflow.plot_gaintable.print_exc")
def test_should_execute_wrapped_function_safely(
    print_exception_mock, logger_mock
):
    def add(a, b=100, c=10):
        return a + b + c

    wrapped_function = safe(add)

    result = wrapped_function(20, c=30)

    logger_mock.error.assert_not_called()
    print_exception_mock.assert_not_called()

    assert result == 150


@patch("ska_sdp_instrumental_calibration.workflow.plot_gaintable.logger")
@patch("ska_sdp_instrumental_calibration.workflow.plot_gaintable.print_exc")
def test_should_catch_exceptions_of_wrapped_function(
    print_exception_mock, logger_mock
):
    def unsafe_function(a, b=4):
        raise Exception(f"Got a: {a} and b: {b}")

    wrapped_function = safe(unsafe_function)

    wrapped_function(10)

    logger_mock.error.assert_called_once_with(
        "Caught exception in function %s: %s",
        "unsafe_function",
        "Got a: 10 and b: 4",
    )
    print_exception_mock.assert_called_once_with()


def test_should_map_channel_to_frequency():
    freq_plot = PlotGaintableFrequency()
    frequency = [0.1, 0.2, 0.3]
    channel = np.arange(len(frequency))
    ch_to_freq_map = freq_plot._primary_sec_ax_mapper(frequency, channel)
    freq_to_ch_map = freq_plot._primary_sec_ax_mapper(
        frequency, channel, reverse=True
    )

    np.testing.assert_allclose(ch_to_freq_map(channel), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(freq_to_ch_map(frequency), [0, 1, 2])


def test_should_map_time_to_time_index():
    freq_plot = PlotGaintableTime()
    time_data = [0.1, 0.2, 0.3]
    time_index = np.arange(len(time_data))
    time_to_index_map = freq_plot._primary_sec_ax_mapper(time_data, time_index)
    index_to_time_map = freq_plot._primary_sec_ax_mapper(
        time_data, time_index, reverse=True
    )

    np.testing.assert_allclose(time_to_index_map(time_data), [0, 1, 2])
    np.testing.assert_allclose(index_to_time_map(time_index), [0.1, 0.2, 0.3])
