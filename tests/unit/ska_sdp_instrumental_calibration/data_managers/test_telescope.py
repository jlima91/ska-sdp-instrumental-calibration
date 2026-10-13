import numpy as np
from dask.base import tokenize
from mock import patch

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
