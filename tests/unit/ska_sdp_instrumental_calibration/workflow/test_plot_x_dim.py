import numpy as np
from mock import ANY, MagicMock, call, patch

from ska_sdp_instrumental_calibration.workflow.plot_x_dim import (
    XDim_Frequency,
    XDim_Time,
    channel_frequency_mapper,
)


def test_should_map_channel_to_frequency():
    frequency = [0.1, 0.2, 0.3]
    channel = np.arange(len(frequency))
    ch_to_freq_map = channel_frequency_mapper(frequency)
    freq_to_ch_map = channel_frequency_mapper(frequency, reverse=True)

    np.testing.assert_allclose(ch_to_freq_map(channel), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(freq_to_ch_map(frequency), [0, 1, 2])


@patch(
    "ska_sdp_instrumental_calibration.workflow.plot_x_dim."
    "channel_frequency_mapper"
)
def test_should_handle_frequency_x_dim(chanel_freq_mapp_mock):
    frequency = np.array([1e6, 2e6, 3e6])
    mock_gaintable = MagicMock(name="gaintable")
    mock_gaintable.frequency = frequency
    mock_gain = MagicMock(name="gain")
    mock_gaintable.gain = mock_gain
    chanel_freq_mapp_mock.side_effect = ["ch_to_freq_map", "freq_to_ch_map"]

    p_ax_mock = MagicMock(name="primary ax")
    s_ax_mock = MagicMock(name="secondary ax")
    s_ax_mock.secondary_xaxis.return_value = s_ax_mock

    gains = XDim_Frequency.gain(mock_gaintable, 1)

    mock_gain.isel.assert_called_once_with(time=0, antenna=1)
    assert gains == mock_gain.isel.return_value

    data, support = XDim_Frequency.data(mock_gaintable)

    np.testing.assert_allclose(data, [0, 1, 2])
    np.testing.assert_allclose(support, [1, 2, 3])

    XDim_Frequency.x_axis(p_ax_mock, s_ax_mock, frequency)

    p_ax_mock.set_xlabel.assert_called_once_with("Channel")
    s_ax_mock.set_xlabel.assert_called_once_with("Frequency [MHz]")
    s_ax_mock.secondary_xaxis.assert_called_once_with(
        "top",
        functions=ANY,
    )
    _, kwarg = s_ax_mock.secondary_xaxis.call_args
    assert kwarg["functions"] == ("ch_to_freq_map", "freq_to_ch_map")
    chanel_freq_mapp_mock.assert_has_calls(
        [call(frequency), call(frequency, reverse=True)]
    )


def test_should_handle_time_x_dim():
    time = np.array([1, 2, 3])
    mock_gaintable = MagicMock(name="gaintable")
    mock_gaintable.time = time
    mock_gain = MagicMock(name="gain")
    mock_gaintable.gain = mock_gain

    p_ax_mock = MagicMock(name="primary ax")
    s_ax_mock = MagicMock(name="secondary ax")

    gains = XDim_Time.gain(mock_gaintable, 1)

    mock_gain.isel.assert_called_once_with(frequency=0, antenna=1)
    assert gains == mock_gain.isel.return_value

    data, support = XDim_Time.data(mock_gaintable)

    np.testing.assert_allclose(data, [1, 2, 3])
    assert support is None

    XDim_Time.x_axis(p_ax_mock, s_ax_mock, time)

    p_ax_mock.set_xlabel.assert_called_once_with("Time (S)")
    s_ax_mock.set_xlabel.assert_not_called()
    s_ax_mock.secondary_xaxis.assert_not_called()
