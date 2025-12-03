import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.stages._utils import (
    _create_path_tree,
    get_gaintables_path,
    get_plots_path,
    get_visibilities_path,
    parse_reference_antenna,
)


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


@patch("ska_sdp_instrumental_calibration.stages._utils.Path")
def test_should_create_path_tree(path_mock):
    path_mock.return_value = path_mock
    _create_path_tree("/output/path")

    path_mock.assert_called_once_with("/output/path")
    path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch("ska_sdp_instrumental_calibration.stages._utils._create_path_tree")
def test_should_get_gaintables_path(create_path_tree_mock):
    result = get_gaintables_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with("/output/gaintables/prefix")

    assert result == "/output/gaintables/prefix"


@patch("ska_sdp_instrumental_calibration.stages._utils._create_path_tree")
def test_should_get_plots_path(create_path_tree_mock):
    result = get_plots_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with("/output/plots/prefix")

    assert result == "/output/plots/prefix"


@patch("ska_sdp_instrumental_calibration.stages._utils._create_path_tree")
def test_should_get_visibilities_path(create_path_tree_mock):
    result = get_visibilities_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with(
        "/output/visibilities/prefix"
    )

    assert result == "/output/visibilities/prefix"
