import numpy as np
import pytest
import xarray as xr
from mock import MagicMock

from ska_sdp_instrumental_calibration.xarray_processors._utils import (
    parse_antenna,
    with_chunks,
)


def test_should_chunk_xarray_object_with_valid_chunks():
    data = xr.DataArray(np.arange(12).reshape(4, 3), dims=["a", "b"])

    chunks = {"a": 2, "c": 4}

    new_data = with_chunks(data, chunks)

    assert dict(new_data.chunksizes) == {"a": (2, 2), "b": (3,)}


def test_should_parse_reference_antenna():
    refant = "LOWBD2_344"
    antennas = ["LOWBD2_344", "LOWBD2_345", "LOWBD2_346", "LOWBD2_347"]
    dims = "id"
    coords = {"id": np.arange(4)}
    ant_names = xr.DataArray(antennas, dims=dims, coords=coords)

    output = parse_antenna(refant, ant_names, 4)

    assert output == 0


def test_should_raise_error_when_ref_ant_is_invalid():
    refant = "ANTENNA-1"
    antennas = ["LOWBD2_344", "LOWBD2_345", "LOWBD2_346", "LOWBD2_347"]
    dims = "id"
    coords = {"id": np.arange(4)}
    ant_names = xr.DataArray(antennas, dims=dims, coords=coords)

    with pytest.raises(ValueError) as error:
        parse_antenna(refant, ant_names, 4)
    assert str(error.value) == "Reference antenna name is not valid"


def test_should_raise_error_when_antenna_index_is_invalid():
    refant = 10
    antnames_mock = MagicMock(name="gaintable")
    with pytest.raises(ValueError) as error:
        parse_antenna(refant, antnames_mock, 5)
    assert str(error.value) == "Reference antenna index is not valid"
