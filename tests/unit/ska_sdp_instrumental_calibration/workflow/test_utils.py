from mock import MagicMock, Mock, call, patch
from numpy import array

from ska_sdp_instrumental_calibration.workflow.utils import (
    plot_gaintable,
    subplot_gaintable,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.utils.np.split",
    return_value=["first set of stations", "second set of stations"],
)
@patch("ska_sdp_instrumental_calibration.workflow.utils.subplot_gaintable")
def test_should_plot_the_gaintable(subplot_gaintable_mock, split_mock):
    gaintable_mock = MagicMock(name="gaintable")

    gaintable_mock.stack.return_value = gaintable_mock
    gaintable_mock.pol.data = [("X", "X"), ("Y", "Y")]
    gaintable_mock.assign_coords.return_value = gaintable_mock

    plot_gaintable(
        gaintable_mock, "/some/path", figure_title="some_title"
    ).compute()

    gaintable_mock.stack.assert_called_with(pol=("receptor1", "receptor2"))
    gaintable_mock.assign_coords.assert_called_with({"pol": ["XX", "YY"]})
    subplot_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "first set of stations",
                "/some/path",
                3,
                3,
                "some_title",
            ),
            call(
                gaintable_mock,
                "second set of stations",
                "/some/path",
                3,
                3,
                "some_title",
            ),
        ]
    )


@patch("ska_sdp_instrumental_calibration.workflow.utils.plt")
@patch("ska_sdp_instrumental_calibration.workflow.utils.np")
def test_should_create_subplots(numpy_mock, plt_mock):
    stations = [1, 2, 3, 4]
    n_rows = 2
    n_cols = 2
    figure_title = "figure title"
    transposed_data = [
        [Mock(name="xx1"), Mock(name="xx1")],
        [Mock(name="yy2"), Mock(name="yy2")],
    ]

    gaintable_mock = MagicMock(name="gaintable")
    amp_mock = MagicMock(name="amp")
    phase_mock = MagicMock(name="phase")
    frequency_mock = MagicMock(name="frequency")

    gain_mock = Mock(name="gain")
    fig_mock = Mock(name="fig mock")
    subfig_mock = Mock(name="subfig")
    amp_axis_mock = Mock(name="amp axis")
    phase_axis_mock = Mock(name="phase axis")

    def secondary_axis_side_effect(pos, functions):
        assert pos == "top"
        assert len(functions) == 2
        assert callable(functions[0])
        functions[0](1)
        functions[1](1)
        assert callable(functions[1])
        return phase_axis_mock

    phase_axis_mock.secondary_xaxis = Mock(
        name="secondary_axix", side_effect=secondary_axis_side_effect
    )
    handles_mock = Mock(name="handles")
    labels_mock = Mock(name="labels")

    frequency_mock.__truediv__.return_value = frequency_mock
    gaintable_mock.frequency = frequency_mock
    gaintable_mock.pol.values = ["XX", "YY"]
    gaintable_mock.gain.isel.return_value = gain_mock
    gaintable_mock.stack.return_value = gaintable_mock
    numpy_mock.abs.return_value = amp_mock
    numpy_mock.angle.return_value = phase_mock
    numpy_mock.arange.return_value = [1, 2]
    amp_mock.T = transposed_data
    phase_mock.T = transposed_data
    plt_mock.figure.return_value = fig_mock
    fig_mock.subfigures.return_value = array(
        [
            [subfig_mock, subfig_mock],
            [subfig_mock, subfig_mock],
        ]
    )
    subfig_mock.subplots.return_value = [phase_axis_mock, amp_axis_mock]
    amp_axis_mock.get_legend_handles_labels.return_value = [
        handles_mock,
        labels_mock,
    ]

    subplot_gaintable(
        gaintable_mock,
        stations,
        "/some/path/file",
        n_rows,
        n_cols,
        figure_title,
    )

    plt_mock.figure.assert_called_once_with(
        layout="constrained", figsize=(18, 18)
    )
    fig_mock.subfigures.assert_called_once_with(2, 2)

    gaintable_mock.gain.isel.assert_has_calls(
        [
            call(time=0, antenna=1),
            call(time=0, antenna=2),
            call(time=0, antenna=3),
            call(time=0, antenna=4),
        ]
    )

    numpy_mock.interp.assert_has_calls(
        [
            call(
                1,
                [
                    1,
                    2,
                ],
                frequency_mock,
            ),
            call(
                1,
                frequency_mock,
                [
                    1,
                    2,
                ],
            ),
            call(
                1,
                [
                    1,
                    2,
                ],
                frequency_mock,
            ),
            call(
                1,
                frequency_mock,
                [
                    1,
                    2,
                ],
            ),
            call(
                1,
                [
                    1,
                    2,
                ],
                frequency_mock,
            ),
            call(
                1,
                frequency_mock,
                [
                    1,
                    2,
                ],
            ),
            call(
                1,
                [
                    1,
                    2,
                ],
                frequency_mock,
            ),
            call(
                1,
                frequency_mock,
                [
                    1,
                    2,
                ],
            ),
        ]
    )

    numpy_mock.abs.assert_has_calls(
        [call(gain_mock), call(gain_mock), call(gain_mock), call(gain_mock)]
    )

    numpy_mock.angle.assert_has_calls(
        [
            call(gain_mock, deg=True),
            call(gain_mock, deg=True),
            call(gain_mock, deg=True),
            call(gain_mock, deg=True),
        ]
    )

    subfig_mock.subplots.assert_has_calls(
        [
            call(2, 1, sharex=True),
            call(2, 1, sharex=True),
            call(2, 1, sharex=True),
            call(2, 1, sharex=True),
        ]
    )

    amp_axis_mock.set_ylabel.assert_has_calls(
        [
            call("Amplitude"),
            call("Amplitude"),
            call("Amplitude"),
            call("Amplitude"),
        ]
    )

    amp_axis_mock.set_xlabel.assert_has_calls(
        [call("Channel"), call("Channel"), call("Channel"), call("Channel")]
    )

    amp_axis_mock.set_xlabel.assert_has_calls(
        [call("Channel"), call("Channel"), call("Channel"), call("Channel")]
    )

    phase_axis_mock.set_ylabel.assert_has_calls(
        [
            call("Phase (degree)"),
            call("Phase (degree)"),
            call("Phase (degree)"),
            call("Phase (degree)"),
        ]
    )

    phase_axis_mock.set_ylim.assert_has_calls(
        [
            call([-180, 180]),
            call([-180, 180]),
            call([-180, 180]),
            call([-180, 180]),
        ]
    )

    subfig_mock.suptitle.assert_has_calls(
        [
            call("Station - 1", fontsize="large"),
            call("Station - 2", fontsize="large"),
            call("Station - 3", fontsize="large"),
            call("Station - 4", fontsize="large"),
        ]
    )

    amp_axis_mock.scatter.assert_has_calls(
        [
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
        ]
    )

    phase_axis_mock.scatter.assert_has_calls(
        [
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
            call([1, 2], transposed_data[0], label="XX"),
            call([1, 2], transposed_data[1], label="YY"),
        ]
    )

    fig_mock.suptitle.assert_called_once_with(
        "figure title Solutions", fontsize="x-large"
    )

    fig_mock.legend.assert_called_once_with(
        handles_mock, labels_mock, loc="outside upper right"
    )

    fig_mock.savefig.assert_called_once_with(
        "/some/path/file-amp-phase_freq1-4.png"
    )

    plt_mock.close.assert_called_once_with()


@patch("ska_sdp_instrumental_calibration.workflow.utils.plt")
@patch("ska_sdp_instrumental_calibration.workflow.utils.np")
def test_should_plot_when_stations_are_less_than_subplot_capacity(
    numpy_mock, plt_mock
):
    stations = [1, 2, 3]
    n_rows = 2
    n_cols = 2
    figure_title = "figure title"
    transposed_data = [
        [Mock(name="xx1"), Mock(name="xx1")],
        [Mock(name="yy2"), Mock(name="yy2")],
    ]

    gaintable_mock = MagicMock(name="gaintable")
    amp_mock = MagicMock(name="amp")
    phase_mock = MagicMock(name="phase")
    frequency_mock = MagicMock(name="frequency")

    gain_mock = Mock(name="gain")
    fig_mock = Mock(name="fig mock")
    subfig_mock = Mock(name="subfig")
    amp_axis_mock = Mock(name="amp axis")
    phase_axis_mock = Mock(name="phase axis")
    handles_mock = Mock(name="handles")
    labels_mock = Mock(name="labels")

    frequency_mock.__truediv__.return_value = frequency_mock
    gaintable_mock.frequency = frequency_mock
    gaintable_mock.pol.values = ["XX", "YY"]
    gaintable_mock.gain.isel.return_value = gain_mock
    gaintable_mock.stack.return_value = gaintable_mock
    numpy_mock.abs.return_value = amp_mock
    numpy_mock.angle.return_value = phase_mock
    numpy_mock.arange.return_value = [1, 2]
    amp_mock.T = transposed_data
    phase_mock.T = transposed_data
    plt_mock.figure.return_value = fig_mock
    fig_mock.subfigures.return_value = array(
        [
            [subfig_mock, subfig_mock],
            [subfig_mock, subfig_mock],
        ]
    )
    subfig_mock.subplots.return_value = [phase_axis_mock, amp_axis_mock]
    amp_axis_mock.get_legend_handles_labels.return_value = [
        handles_mock,
        labels_mock,
    ]

    subplot_gaintable(
        gaintable_mock,
        stations,
        "/some/path/file",
        n_rows,
        n_cols,
        figure_title,
    )
