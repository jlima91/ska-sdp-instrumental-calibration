import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, Mock, call, patch
from numpy import array, testing

from ska_sdp_instrumental_calibration.workflow.utils import (
    create_path_tree,
    ecef_to_lla,
    get_gaintables_path,
    get_plots_path,
    get_visibilities_path,
    parse_reference_antenna,
    plot_all_stations,
    plot_gaintable,
    safe,
    subplot_gaintable,
    with_chunks,
)


@patch("ska_sdp_instrumental_calibration.workflow.utils.logger")
@patch("ska_sdp_instrumental_calibration.workflow.utils.print_exc")
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


@patch("ska_sdp_instrumental_calibration.workflow.utils.logger")
@patch("ska_sdp_instrumental_calibration.workflow.utils.print_exc")
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


@patch("ska_sdp_instrumental_calibration.workflow.utils.subplot_gaintable")
def test_should_plot_the_gaintable(subplot_gaintable_mock):
    gaintable_mock = MagicMock(name="gaintable")

    gaintable_mock.stack.return_value = gaintable_mock
    gaintable_mock.pol.data = [("X", "X"), ("Y", "Y")]
    gaintable_mock.assign_coords.return_value = gaintable_mock
    gaintable_mock.configuration.names = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    plot_gaintable(
        gaintable_mock,
        "/some/path",
        figure_title="some_title",
        fixed_axis=False,
    ).compute()

    gaintable_mock.stack.assert_called_with(pol=("receptor1", "receptor2"))
    gaintable_mock.assign_coords.assert_called_with({"pol": ["XX", "YY"]})

    first_call_actual = subplot_gaintable_mock.call_args_list[0].kwargs
    testing.assert_allclose(first_call_actual["gaintable"], gaintable_mock)
    testing.assert_string_equal(first_call_actual["path_prefix"], "/some/path")
    testing.assert_allclose(first_call_actual["n_rows"], 3)
    testing.assert_allclose(first_call_actual["n_cols"], 3)
    testing.assert_string_equal(
        first_call_actual["figure_title"], "some_title"
    )
    testing.assert_allclose(first_call_actual["fixed_axis"], False)
    testing.assert_allclose(
        first_call_actual["stations"], [1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    second_call_actual = subplot_gaintable_mock.call_args_list[1].kwargs
    testing.assert_allclose(second_call_actual["gaintable"], gaintable_mock)
    testing.assert_string_equal(
        second_call_actual["path_prefix"], "/some/path"
    )
    testing.assert_allclose(second_call_actual["n_rows"], 3)
    testing.assert_allclose(second_call_actual["n_cols"], 3)
    testing.assert_string_equal(
        second_call_actual["figure_title"], "some_title"
    )
    testing.assert_allclose(second_call_actual["fixed_axis"], False)
    testing.assert_allclose(second_call_actual["stations"], [10])


@patch("ska_sdp_instrumental_calibration.workflow.utils.subplot_gaintable")
@patch("ska_sdp_instrumental_calibration.workflow.utils.plot_all_stations")
def test_should_plot_all_station_plot_for_bp_solution(
    plot_all_staions_mock, subplot_gaintable_mock
):
    gaintable_mock = MagicMock(name="gaintable")

    gaintable_mock.stack.return_value = gaintable_mock
    gaintable_mock.pol.data = [("X", "X"), ("Y", "Y")]
    gaintable_mock.assign_coords.return_value = gaintable_mock

    plot_gaintable(
        gaintable_mock,
        "/some/path",
        figure_title="Bandpass",
        fixed_axis=False,
        all_station_plot=True,
    ).compute()

    gaintable_mock.stack.assert_called_once_with(
        pol=("receptor1", "receptor2")
    )
    gaintable_mock.assign_coords.assert_called_once_with({"pol": ["XX", "YY"]})
    plot_all_staions_mock.assert_called_once_with(gaintable_mock, "/some/path")


@patch("ska_sdp_instrumental_calibration.workflow.utils.subplot_gaintable")
def test_should_plot_only_parallel_hand_pols(subplot_gaintable_mock):
    gaintable_mock = MagicMock(name="gaintable")

    gaintable_mock.stack.return_value = gaintable_mock
    gaintable_mock.pol.data = [("X", "X"), ("Y", "Y")]
    gaintable_mock.assign_coords.return_value = gaintable_mock

    plot_gaintable(
        gaintable_mock,
        "/some/path",
        figure_title="Channel Rotation Measure",
        fixed_axis=False,
        drop_cross_pols=True,
    ).compute()

    gaintable_mock.stack.assert_called_once_with(
        pol=("receptor1", "receptor2")
    )
    gaintable_mock.assign_coords.assert_called_once_with({"pol": ["XX", "YY"]})
    gaintable_mock.sel.assert_called_once_with(pol=["XX", "YY"])


@patch("ska_sdp_instrumental_calibration.workflow.utils.np.linspace")
@patch("ska_sdp_instrumental_calibration.workflow.utils.np.abs")
@patch("ska_sdp_instrumental_calibration.workflow.utils.plt")
@patch("ska_sdp_instrumental_calibration.workflow.utils.cm")
def test_should_create_amp_vs_freq_plot_for_all_stations(
    cm_mock, plt_mock, abs_mock, linspace_mock
):
    gaintable_mock = MagicMock(name="gaintable")
    station_1_mock = Mock(name="station1")
    station_2_mock = Mock(name="station2")
    amp_mock = Mock(name="amp")
    frequency_mock = Mock(name="frequency")
    fig_mock = Mock(name="fig")
    ax_mock = Mock(name="ax")
    cmap_mock = Mock(name="cmap")
    sm_mock = Mock(name="sm")
    norm_mock = Mock(name="norm")
    colorbar_mock = Mock(name="colorbar")

    gaintable_mock.frequency = frequency_mock
    gaintable_mock.antenna.size = 10
    abs_mock.return_value = amp_mock
    amp_mock.sel.return_value = [station_1_mock, station_2_mock]
    plt_mock.subplots.return_value = [fig_mock, ax_mock]
    plt_mock.get_cmap.return_value = cmap_mock
    cm_mock.ScalarMappable.return_value = sm_mock
    plt_mock.Normalize.return_value = norm_mock
    linspace_mock.return_value = [0, 1, 2]
    fig_mock.colorbar.return_value = colorbar_mock

    plot_all_stations(gaintable_mock, "path-prefix")

    plt_mock.get_cmap.assert_called_once_with("viridis", 10)
    plt_mock.Normalize.assert_called_once_with(vmin=0, vmax=9)
    cm_mock.ScalarMappable.assert_called_once_with(
        norm=norm_mock, cmap=cmap_mock
    )

    plt_mock.subplots.assert_has_calls(
        [call(figsize=(10, 10)), call(figsize=(10, 10))]
    )
    ax_mock.plot.assert_has_calls(
        [
            call(frequency_mock, station_1_mock, color=cmap_mock(1)),
            call(frequency_mock, station_2_mock, color=cmap_mock(2)),
            call(frequency_mock, station_1_mock, color=cmap_mock(1)),
            call(frequency_mock, station_2_mock, color=cmap_mock(2)),
        ]
    )

    amp_mock.sel.assert_has_calls([call(pol="XX"), call(pol="YY")])

    ax_mock.set_title.assert_has_calls(
        [
            call("All station Amp vs Freq for pol XX"),
            call("All station Amp vs Freq for pol YY"),
        ]
    )
    ax_mock.set_xlabel.assert_has_calls([call("Freq [HZ]"), call("Freq [HZ]")])
    ax_mock.set_ylabel.assert_has_calls([call("Amp"), call("Amp")])
    fig_mock.savefig.assert_has_calls(
        [
            call(
                "path-prefix-all_station_amp_vs_freq_XX.png",
                bbox_inches="tight",
            ),
            call(
                "path-prefix-all_station_amp_vs_freq_YY.png",
                bbox_inches="tight",
            ),
        ]
    )

    linspace_mock.assert_has_calls(
        [call(0, 10, 11, dtype=int), call(0, 10, 11, dtype=int)]
    )
    fig_mock.colorbar.assert_has_calls(
        [
            call(sm_mock, ax=ax_mock, ticks=[0, 1, 2]),
            call(sm_mock, ax=ax_mock, ticks=[0, 1, 2]),
        ]
    )

    colorbar_mock.ax.set_title.assert_has_calls(
        [
            call("Stations", loc="center", fontsize=10),
            call("Stations", loc="center", fontsize=10),
        ]
    )

    plt_mock.close.assert_has_calls([call(fig_mock), call(fig_mock)])


@patch("ska_sdp_instrumental_calibration.workflow.utils.plt")
@patch("ska_sdp_instrumental_calibration.workflow.utils.np")
def test_should_create_subplots(numpy_mock, plt_mock):
    n_rows = 2
    n_cols = 2
    figure_title = "figure title"
    transposed_data = [
        [Mock(name="xx1"), Mock(name="xx1")],
        [Mock(name="yy2"), Mock(name="yy2")],
    ]

    stations = Mock(name="stations")
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

    stations.id = [1, 2, 3, 4]
    stations.values = ["a", "b", "c", "d"]
    stations.size = 4
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
        False,
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
            call("Station - a", fontsize="large"),
            call("Station - b", fontsize="large"),
            call("Station - c", fontsize="large"),
            call("Station - d", fontsize="large"),
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
        "/some/path/file-amp-phase_freq-a-d.png"
    )

    plt_mock.close.assert_called_once_with()


@patch("ska_sdp_instrumental_calibration.workflow.utils.plt")
@patch("ska_sdp_instrumental_calibration.workflow.utils.np")
def test_should_plot_when_stations_are_less_than_subplot_capacity(
    numpy_mock, plt_mock
):
    n_rows = 2
    n_cols = 2
    figure_title = "figure title"
    transposed_data = [
        [Mock(name="xx1"), Mock(name="xx1")],
        [Mock(name="yy2"), Mock(name="yy2")],
    ]

    stations_mock = Mock(name="stations")
    gaintable_mock = MagicMock(name="gaintable")
    amp_mock = MagicMock(name="amp")
    phase_mock = MagicMock(name="phase")
    frequency_mock = MagicMock(name="frequency")

    stations_mock.id = [11, 12, 13]
    stations_mock.values = ["x", "y", "z"]
    stations_mock.size = 3

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
        stations_mock,
        "/some/path/file",
        n_rows,
        n_cols,
        True,
        figure_title,
    )

    amp_axis_mock.set_ylim.assert_called_with([0, 1])
    subfig_mock.subplots.assert_has_calls(
        [
            call(2, 1, sharex=True),
            call(2, 1, sharex=True),
            call(2, 1, sharex=True),
        ]
    )


def test_should_convert_earth_centric_coordinates_to_geodetic():
    x = np.array([-5133977.79947732, -5133977.79947732])
    y = np.array([10168886.79974105, -10168886.79974105])
    z = np.array([-5723265.90488199, -5723265.90488199])

    lat, lng, alt = ecef_to_lla(x, y, z)

    np.testing.assert_allclose(lat, np.array([-26.753052, -26.753052]))
    np.testing.assert_allclose(lng, np.array([116.787894, 243.212106]))
    np.testing.assert_allclose(alt, np.array([6374502.632896, 6374502.632896]))


def test_should_parse_reference_antenna():
    refant = "LOWBD2_344"
    antennas = ["LOWBD2_344", "LOWBD2_345", "LOWBD2_346", "LOWBD2_347"]
    dims = "id"
    coords = {"id": np.arange(4)}
    gaintable_mock = MagicMock(name="gaintable")
    ant_names = xr.DataArray(antennas, dims=dims, coords=coords)
    gaintable_mock.configuration.names = ant_names
    output = parse_reference_antenna(refant, gaintable_mock)

    assert output == 0


def test_should_raise_error_when_ref_ant_is_invalid():
    refant = "ANTENNA-1"
    antennas = ["LOWBD2_344", "LOWBD2_345", "LOWBD2_346", "LOWBD2_347"]
    dims = "id"
    coords = {"id": np.arange(4)}
    gaintable_mock = MagicMock(name="gaintable")
    ant_names = xr.DataArray(antennas, dims=dims, coords=coords)
    gaintable_mock.configuration.names = ant_names
    with pytest.raises(ValueError) as error:
        parse_reference_antenna(refant, gaintable_mock)
    assert str(error.value) == "Reference antenna name is not valid"


def test_should_raise_error_when_antenna_index_is_invalid():
    refant = 10
    gaintable_mock = MagicMock(name="gaintable")
    gaintable_mock.antenna.size = 5
    with pytest.raises(ValueError) as error:
        parse_reference_antenna(refant, gaintable_mock)
    assert str(error.value) == "Reference antenna index is not valid"


def test_should_chunk_xarray_object_with_valid_chunks():
    data = xr.DataArray(np.arange(12).reshape(4, 3), dims=["a", "b"])

    chunks = {"a": 2, "c": 4}

    new_data = with_chunks(data, chunks)

    assert dict(new_data.chunksizes) == {"a": (2, 2), "b": (3,)}


@patch("ska_sdp_instrumental_calibration.workflow.utils.Path")
def test_should_create_path_tree(path_mock):
    path_mock.return_value = path_mock
    create_path_tree("/output/path")

    path_mock.assert_called_once_with("/output/path")
    path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch("ska_sdp_instrumental_calibration.workflow.utils.create_path_tree")
def test_should_get_plots_path(create_path_tree_mock):
    result = get_plots_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with("/output/plots/prefix")

    assert result == "/output/plots/prefix"


@patch("ska_sdp_instrumental_calibration.workflow.utils.create_path_tree")
def test_should_get_gaintables_path(create_path_tree_mock):
    result = get_gaintables_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with("/output/gaintables/prefix")

    assert result == "/output/gaintables/prefix"


@patch("ska_sdp_instrumental_calibration.workflow.utils.create_path_tree")
def test_should_get_visibilities_path(create_path_tree_mock):
    result = get_visibilities_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with(
        "/output/visibilities/prefix"
    )

    assert result == "/output/visibilities/prefix"
