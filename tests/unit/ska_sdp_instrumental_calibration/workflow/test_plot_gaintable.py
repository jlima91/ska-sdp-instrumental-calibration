import numpy as np
from mock import ANY, MagicMock, call, patch

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
    gain_plotter = PlotGaintableFrequency()
    frequency = [0.1, 0.2, 0.3]
    channel = np.arange(len(frequency))
    ch_to_freq_map = gain_plotter._primary_sec_ax_mapper(frequency, channel)
    freq_to_ch_map = gain_plotter._primary_sec_ax_mapper(
        frequency, channel, reverse=True
    )

    np.testing.assert_allclose(ch_to_freq_map(channel), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(freq_to_ch_map(frequency), [0, 1, 2])


def test_should_map_time_to_time_index():
    gain_plotter = PlotGaintableTime()
    time_data = [0.1, 0.2, 0.3]
    time_index = np.arange(len(time_data))
    time_to_index_map = gain_plotter._primary_sec_ax_mapper(
        time_data, time_index
    )
    index_to_time_map = gain_plotter._primary_sec_ax_mapper(
        time_data, time_index, reverse=True
    )

    np.testing.assert_allclose(time_to_index_map(time_data), [0, 1, 2])
    np.testing.assert_allclose(index_to_time_map(time_index), [0.1, 0.2, 0.3])


@patch("ska_sdp_instrumental_calibration.workflow.plot_gaintable.np")
def test_should_plot_gaintable_for_freq(np_mock):
    gaintable = MagicMock(name="gaintable")
    gaintable.stack.return_value = gaintable
    gaintable.assign_coords.return_value = gaintable
    gaintable.swap_dims.return_value = gaintable

    jones_solution_mock = MagicMock(name="jones_solution_mock")
    jones_solution_mock.data = [("X", "X"), ("X", "Y"), ("Y", "Y")]
    phase_gain_mock = MagicMock(name="gain_phase")
    amp_gain_mock = MagicMock(name="gain")
    gain_mock = MagicMock(name="gain")
    gaintable.gain = gain_mock
    gain_mock.copy.return_value = phase_gain_mock
    mock_facet_phase = MagicMock(name="facet_plot_phase")
    mock_facet_amp = MagicMock(name="facet_plot_amp")
    amp_gain_mock.plot.scatter.return_value = mock_facet_amp
    phase_gain_mock.plot.scatter.return_value = mock_facet_phase
    np_mock.abs.return_value = amp_gain_mock
    gaintable.__getitem__.return_value = jones_solution_mock

    plotter = PlotGaintableFrequency(path_prefix="path/to/save")

    delayed_plot = plotter.plot(gaintable, figure_title="Plot Title")
    delayed_plot.compute()

    gaintable.stack.assert_called_once_with(
        Jones_Solutions=("receptor1", "receptor2")
    )

    gaintable.assign_coords.assert_called_once_with(
        {"Jones_Solutions": ["J_XX", "J_XY", "J_YY"]}
    )

    gaintable.swap_dims.assert_has_calls(
        [call({"antenna": "Station"}), call({"frequency": "Channel"})]
    )
    amp_gain_mock.plot.scatter.assert_has_calls(
        [
            call(
                x="Channel",
                hue="Jones_Solutions",
                col="Station",
                col_wrap=5,
                add_legend=True,
                add_colorbar=False,
                sharex=False,
                edgecolors="none",
                aspect=1.5,
                s=8,
                ylim=None,
            )
        ]
    )

    phase_gain_mock.plot.scatter.assert_has_calls(
        [
            call(
                x="Channel",
                hue="Jones_Solutions",
                col="Station",
                col_wrap=5,
                add_legend=True,
                add_colorbar=False,
                sharex=False,
                edgecolors="none",
                aspect=1.5,
                s=8,
                ylim=None,
            )
        ]
    )

    mock_facet_phase.fig.suptitle.assert_called_once_with(
        "Plot Title Solutions (Phase)", fontsize="x-large", y=1.08
    )
    mock_facet_phase.fig.tight_layout.assert_called_once()
    mock_facet_phase.fig.savefig.assert_called_once_with(
        "path/to/save-phase-freq.png", bbox_inches="tight"
    )

    mock_facet_amp.fig.suptitle.assert_called_once_with(
        "Plot Title Solutions (Amplitude)", fontsize="x-large", y=1.08
    )
    mock_facet_amp.fig.tight_layout.assert_called_once()
    mock_facet_amp.fig.savefig.assert_called_once_with(
        "path/to/save-amp-freq.png", bbox_inches="tight"
    )


@patch("ska_sdp_instrumental_calibration.workflow.plot_gaintable.np")
def test_should_plot_gaintable_for_time(np_mock):
    gaintable = MagicMock(name="gaintable")
    gaintable.stack.return_value = gaintable
    gaintable.assign_coords.return_value = gaintable
    gaintable.swap_dims.return_value = gaintable
    gaintable.assign.return_value = gaintable

    gaintable.time = np.array([1, 2, 3, 4])

    jones_solution_mock = MagicMock(name="jones_solution_mock")
    jones_solution_mock.data = [("X", "X"), ("X", "Y"), ("Y", "Y")]
    phase_gain_mock = MagicMock(name="gain_phase")
    amp_gain_mock = MagicMock(name="gain")
    gain_mock = MagicMock(name="gain")
    gaintable.gain = gain_mock
    gain_mock.copy.return_value = phase_gain_mock
    mock_facet_phase = MagicMock(name="facet_plot_phase")
    mock_facet_amp = MagicMock(name="facet_plot_amp")
    amp_gain_mock.plot.scatter.return_value = mock_facet_amp
    phase_gain_mock.plot.scatter.return_value = mock_facet_phase
    np_mock.abs.return_value = amp_gain_mock
    gaintable.__getitem__.return_value = jones_solution_mock

    plotter = PlotGaintableTime(path_prefix="path/to/save")

    delayed_plot = plotter.plot(gaintable, figure_title="Plot Title")
    delayed_plot.compute()

    gaintable.stack.assert_called_once_with(
        Jones_Solutions=("receptor1", "receptor2")
    )

    gaintable.assign_coords.assert_called_once_with(
        {"Jones_Solutions": ["J_XX", "J_XY", "J_YY"]}
    )

    gaintable.swap_dims.assert_called_once_with({"antenna": "Station"})
    gaintable.assign.assert_called_once_with({"time": ANY})

    amp_gain_mock.plot.scatter.assert_has_calls(
        [
            call(
                x="time",
                hue="Jones_Solutions",
                col="Station",
                col_wrap=5,
                add_legend=True,
                add_colorbar=False,
                sharex=False,
                edgecolors="none",
                aspect=1.5,
                s=8,
                ylim=None,
            )
        ]
    )

    phase_gain_mock.plot.scatter.assert_has_calls(
        [
            call(
                x="time",
                hue="Jones_Solutions",
                col="Station",
                col_wrap=5,
                add_legend=True,
                add_colorbar=False,
                sharex=False,
                edgecolors="none",
                aspect=1.5,
                s=8,
                ylim=None,
            )
        ]
    )

    mock_facet_phase.fig.suptitle.assert_called_once_with(
        "Plot Title Solutions (Phase)", fontsize="x-large", y=1.08
    )
    mock_facet_phase.fig.tight_layout.assert_called_once()
    mock_facet_phase.fig.savefig.assert_called_once_with(
        "path/to/save-phase-time.png", bbox_inches="tight"
    )

    mock_facet_amp.fig.suptitle.assert_called_once_with(
        "Plot Title Solutions (Amplitude)", fontsize="x-large", y=1.08
    )
    mock_facet_amp.fig.tight_layout.assert_called_once()
    mock_facet_amp.fig.savefig.assert_called_once_with(
        "path/to/save-amp-time.png", bbox_inches="tight"
    )
