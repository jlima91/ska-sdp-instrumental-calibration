import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.xarray_processors._utils import (
    parse_antenna,
    simplify_baselines_dim,
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


def test_should_raise_exception_if_unsuported_type_for_antenna():
    with pytest.raises(
        RuntimeError,
        match="Unsupported type for antenna. Only int or string supported",
    ):
        parse_antenna([1, 2, 3], "ant_names", 4)


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


@patch("ska_sdp_instrumental_calibration.xarray_processors._utils.logger")
def test_should_log_warning_and_return_vis_if_baselines_is_none(logger_mock):
    vis = MagicMock(name="vis")

    vis.coords.get.return_value = None

    expected = simplify_baselines_dim(vis)

    vis.coords.get.assert_called_once_with("baselines")
    logger_mock.warning.assert_called_once_with(
        "No baselines coord in dataset. Returning unchanged"
    )
    assert expected == vis


@patch("ska_sdp_instrumental_calibration.xarray_processors._utils.np.arange")
@patch("ska_sdp_instrumental_calibration.xarray_processors._utils.logger")
def test_should_swap_baseline_multi_index_with_coords(
    logger_mock, arange_mock
):
    vis = MagicMock(name="vis")
    vis.coords.get.return_value = "baselines"
    vis.variables.get.return_value = None
    vis_with_baseline_id = MagicMock(name="vis_baseline_id")
    vis.baselines = ["baseline-1", "baseline-2"]
    vis.assign_coords.return_value = vis_with_baseline_id
    vis_with_baseline_id.swap_dims.return_value = vis_with_baseline_id
    vis_with_baseline_id.reset_coords.return_value = vis_with_baseline_id
    arange_mock.return_value = "ARANGE_RETURN"

    expected = simplify_baselines_dim(vis)

    logger_mock.debug.assert_called_once_with(
        "Swapping baselines MultiIndex coord with indices"
    )
    vis.assign_coords.assert_called_once_with(
        baselineid=("baselines", "ARANGE_RETURN")
    )
    vis_with_baseline_id.swap_dims.assert_called_once_with(
        {"baselines": "baselineid"}
    )
    vis_with_baseline_id.reset_coords.assert_called_once_with(
        ("baselines", "antenna1", "antenna2")
    )

    arange_mock.assert_called_once_with(2)
    assert expected == vis_with_baseline_id


@patch("ska_sdp_instrumental_calibration.xarray_processors._utils.logger")
def test_should_swap_baseline_multi_index_with_baseline_coords(logger_mock):
    vis = MagicMock(name="vis")
    vis.coords.get.return_value = "baselines"
    vis.variables.get.return_value = "baselineid"
    vis_with_baseline_id = MagicMock(name="vis_baseline_id")
    vis.baselines = ["baseline-1", "baseline-2"]
    vis.swap_dims.return_value = vis_with_baseline_id
    vis_with_baseline_id.reset_coords.return_value = vis_with_baseline_id

    expected = simplify_baselines_dim(vis)

    logger_mock.debug.assert_called_once_with(
        "Swapping baselines MultiIndex coord with indices"
    )
    vis.swap_dims.assert_called_once_with({"baselines": "baselineid"})
    vis_with_baseline_id.reset_coords.assert_called_once_with(
        ("baselines", "antenna1", "antenna2")
    )
    assert expected == vis_with_baseline_id
