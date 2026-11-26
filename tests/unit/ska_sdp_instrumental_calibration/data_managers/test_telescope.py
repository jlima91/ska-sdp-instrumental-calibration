import numpy as np
import pytest
from dask.base import tokenize
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.telescope import Telescope


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".telescope.read_antenna_response_data"
)
def test_should_tokenize_telescope_from_ms_path_and_model(read_mock):
    token = tokenize(Telescope("a.ms", "oskar_dipole_cos"))

    assert token == tokenize(Telescope("a.ms", "oskar_dipole_cos"))
    assert token != tokenize(Telescope("b.ms", "oskar_dipole_cos"))
    assert token != tokenize(Telescope("a.ms", "hamaker"))


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".telescope.read_antenna_response_data"
)
def test_should_not_tokenize_response_data(read_mock):
    read_mock.side_effect = [
        {"station_names": np.array(["s1"]), "positions": 1},
        {"station_names": np.array(["s1"]), "positions": 2},
    ]

    assert tokenize(Telescope("a.ms", "oskar_dipole_cos")) == tokenize(
        Telescope("a.ms", "oskar_dipole_cos")
    )


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".telescope.read_antenna_response_data"
)
def test_should_forward_directions_to_everybeam_station_response(read_mock):
    read_mock.return_value = {"station_names": np.array(["s1", "s2"])}
    telescope = Telescope("a.ms", "oskar_dipole_cos")
    eb_telescope = MagicMock(name="eb_telescope")
    eb_telescope.station_response.return_value = np.ones((1, 2, 3, 2, 2))
    telescope._telescope = eb_telescope

    direction = np.array([1.0, 0.0, 0.0])
    station0 = np.array([0.0, 1.0, 0.0])
    frequencies = np.array([1.0e8, 1.1e8, 1.2e8])
    scale = np.array([1.0, 2.0, 3.0])

    beams = telescope.station_response(
        10.0, frequencies, direction, station0, scale=scale
    )

    eb_telescope.station_response.assert_called_once()
    kwargs = eb_telescope.station_response.call_args.kwargs
    assert set(kwargs) == {
        "times",
        "station_indexes",
        "freqs",
        "direction",
        "station0_direction",
        "tile0_direction",
    }
    np.testing.assert_array_equal(kwargs["times"], [10.0])
    assert isinstance(kwargs["station_indexes"], np.ndarray)
    np.testing.assert_array_equal(kwargs["station_indexes"], [0, 1])
    np.testing.assert_array_equal(kwargs["freqs"], frequencies)
    np.testing.assert_array_equal(kwargs["direction"], direction)
    np.testing.assert_array_equal(kwargs["station0_direction"], station0)
    np.testing.assert_array_equal(kwargs["tile0_direction"], station0)
    np.testing.assert_array_equal(
        beams[0, 0, :, 0, 0], np.array([1.0, 2.0, 3.0])
    )


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".telescope.read_antenna_response_data_from_msv4"
)
def test_should_tokenize_msv4_telescope_from_response_data_and_model(
    read_mock,
):
    def response_data(position):
        return {
            "station_names": np.array(["s1"]),
            "positions": np.array([position]),
        }

    read_mock.side_effect = lambda ms_xdt: response_data(ms_xdt)

    token = tokenize(Telescope.from_msv4(1.0, "oskar_dipole_cos"))

    assert token == tokenize(Telescope.from_msv4(1.0, "oskar_dipole_cos"))
    assert token != tokenize(Telescope.from_msv4(2.0, "oskar_dipole_cos"))
    assert token != tokenize(Telescope.from_msv4(1.0, "hamaker"))


MODULE = "ska_sdp_instrumental_calibration.data_managers.telescope"
DELAY_DIRECTION = np.array([1.0, 0.5])


def _ms_xdt_with_antennas(names):
    ms_xdt = MagicMock(name="ms_xdt")
    ms_xdt.__getitem__.return_value.ds.antenna_name.values = np.array(names)
    return ms_xdt


def _layout(names, **extra):
    return {"station_names": np.array(names), **extra}


@patch(f"{MODULE}.read_antenna_response_data_from_msv4")
def test_should_create_telescope_for_visibilities_without_eb_ms(read_mock):
    ms_xdt = _ms_xdt_with_antennas(["s0", "s1"])
    read_mock.return_value = _layout(["s0", "s1"])

    telescope = Telescope.for_visibilities(ms_xdt, "oskar_dipole_cos")

    read_mock.assert_called_once_with(ms_xdt)
    assert telescope.response_data is read_mock.return_value


@patch(f"{MODULE}.get_delay_directions", return_value=DELAY_DIRECTION)
@patch(f"{MODULE}.open_processing_set")
@patch(f"{MODULE}.detect_input_type", return_value="msv2")
@patch(f"{MODULE}.read_antenna_response_data")
def test_should_point_msv2_eb_ms_beam_at_the_visibilities_field(
    read_mock, detect_mock, open_mock, delay_mock
):
    ms_xdt = _ms_xdt_with_antennas(["s0", "s1"])
    # Delay directions of all the fields of the MSv2, flattened
    read_mock.return_value = _layout(
        ["s0", "s1"], delay_directions=np.arange(4.0)
    )

    telescope = Telescope.for_visibilities(
        ms_xdt, "oskar_dipole_cos", "eb.ms"
    )

    detect_mock.assert_called_once_with(["eb.ms"])
    read_mock.assert_called_once_with("eb.ms")
    open_mock.assert_not_called()
    delay_mock.assert_called_once_with(ms_xdt)
    np.testing.assert_array_equal(
        telescope.response_data["delay_directions"], DELAY_DIRECTION
    )


@patch(f"{MODULE}.get_delay_directions", return_value=DELAY_DIRECTION)
@patch(f"{MODULE}.open_processing_set")
@patch(f"{MODULE}.detect_input_type", return_value="ps")
@patch(f"{MODULE}.read_station_layout_from_msv4")
def test_should_read_processing_set_eb_ms_layout_from_any_node(
    read_mock, detect_mock, open_mock, delay_mock
):
    ms_xdt = _ms_xdt_with_antennas(["s0", "s1"])
    first_node, second_node = MagicMock(), MagicMock()
    open_mock.return_value.children = {"a": first_node, "b": second_node}
    read_mock.return_value = _layout(["s0", "s1"])

    telescope = Telescope.for_visibilities(
        ms_xdt, "oskar_dipole_cos", "eb.ps.zarr"
    )

    open_mock.assert_called_once_with("eb.ps.zarr")
    read_mock.assert_called_once_with(first_node)
    delay_mock.assert_called_once_with(ms_xdt)
    np.testing.assert_array_equal(
        telescope.response_data["delay_directions"], DELAY_DIRECTION
    )


@patch(f"{MODULE}.open_processing_set")
@patch(f"{MODULE}.detect_input_type", return_value="ps")
def test_should_raise_for_empty_processing_set_eb_ms(detect_mock, open_mock):
    open_mock.return_value.children = {}

    with pytest.raises(ValueError, match="No MSv4 data"):
        Telescope.for_visibilities(
            _ms_xdt_with_antennas(["s0"]), "oskar_dipole_cos", "eb.ps.zarr"
        )


@patch(f"{MODULE}.detect_input_type", return_value="msv2")
@patch(f"{MODULE}.read_antenna_response_data")
def test_should_raise_for_eb_ms_with_other_stations(read_mock, detect_mock):
    ms_xdt = _ms_xdt_with_antennas(["s0", "s1"])
    read_mock.return_value = _layout(["s1", "s0"], delay_directions=None)

    with pytest.raises(ValueError, match="same names in the same order"):
        Telescope.for_visibilities(ms_xdt, "oskar_dipole_cos", "eb.ms")


@patch(f"{MODULE}.get_delay_directions")
@patch(f"{MODULE}.detect_input_type", return_value="msv2")
@patch(f"{MODULE}.read_antenna_response_data")
def test_should_tokenize_eb_ms_telescope_by_delay_direction(
    read_mock, detect_mock, delay_mock
):
    """The same eb_ms gives different telescopes for different fields."""
    ms_xdt = _ms_xdt_with_antennas(["s0"])
    read_mock.side_effect = lambda path: _layout(
        ["s0"], delay_directions=None
    )
    delay_mock.side_effect = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]

    first = Telescope.for_visibilities(ms_xdt, "oskar_dipole_cos", "eb.ms")
    second = Telescope.for_visibilities(ms_xdt, "oskar_dipole_cos", "eb.ms")

    assert tokenize(first) != tokenize(second)
