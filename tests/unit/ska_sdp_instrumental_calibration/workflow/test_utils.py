from mock import MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.workflow.utils import (
    create_plot,
    plot_gaintable,
)


@patch("ska_sdp_instrumental_calibration.workflow.utils.np")
@patch("ska_sdp_instrumental_calibration.workflow.utils.create_plot")
def test_should_plot_the_gaintable(create_plot_mock, numpy_mock):
    gaintable_mock = MagicMock(name="gaintable")
    amp_mock = Mock(name="amplitude")
    phase_mock = Mock(name="phase")
    frequency_mock = Mock(name="frequency")
    gain_mock = Mock(name="gain")
    gain_mock.pol.values = ["XX", "YY"]

    gaintable_mock.frequency = frequency_mock
    gaintable_mock.gain.isel.return_value = gain_mock
    gaintable_mock.stack.return_value = gaintable_mock
    gaintable_mock.pol.data = [("X", "X"), ("Y", "Y")]
    gaintable_mock.assign_coords.return_value = gaintable_mock
    numpy_mock.abs.return_value = amp_mock
    numpy_mock.angle.return_value = phase_mock

    plot_gaintable(gaintable_mock, "/some/path").compute()

    gaintable_mock.stack.assert_called_with(pol=("receptor1", "receptor2"))
    gaintable_mock.assign_coords.assert_called_with({"pol": ["XX", "YY"]})
    gaintable_mock.gain.isel.assert_has_calls([call(time=0, antenna=0)])
    create_plot_mock.assert_has_calls(
        [
            call(
                frequency_mock,
                amp_mock,
                "/some/path-amp_freq.png",
                "Amplitude",
                "Channel Vs Amplitude",
                ["XX", "YY"],
            ),
            call(
                frequency_mock,
                phase_mock,
                "/some/path-phase_freq.png",
                "Phase",
                "Channel Vs Phase",
                ["XX", "YY"],
            ),
        ]
    )


@patch("ska_sdp_instrumental_calibration.workflow.utils.plt")
def test_should_create_a_plot(plt_mock):
    x_data_mock = Mock(name="x")
    y_data_mock = Mock(name="Y")

    create_plot(
        x_data_mock, y_data_mock, "/some/path", "ylable", "title", "label"
    )

    plt_mock.figure.assert_called_once_with(figsize=(10, 8))
    plt_mock.style.use.assert_called_once_with("seaborn-v0_8")
    plt_mock.title.assert_called_once_with("title")
    plt_mock.xlabel.assert_called_once_with("Channel")
    plt_mock.ylabel.assert_called_once_with("ylable")
    plt_mock.plot.assert_called_once_with(
        x_data_mock, y_data_mock, label="label"
    )
    plt_mock.legend.assert_called_once_with()
    plt_mock.savefig.assert_called_once_with("/some/path")
    plt_mock.close.assert_called_once_with()
