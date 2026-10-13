import numpy as np
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
