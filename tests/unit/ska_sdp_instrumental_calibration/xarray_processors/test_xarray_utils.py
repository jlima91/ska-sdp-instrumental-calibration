import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.xarray_processors._utils import (
    parse_antenna,
    simplify_baselines_dim,
    with_chunks,
)


def test_with_chunks_rechunks_dataarray_when_relevant_dim_exists():
    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("time", "frequency"),
    ).chunk({"time": 3, "frequency": 4})

    result = with_chunks(data, {"frequency": 2, "antenna": 1})

    assert result is not data
    assert result.chunksizes["time"] == (3,)
    assert result.chunksizes["frequency"] == (2, 2)


def test_with_chunks_returns_same_dataarray_when_no_relevant_dim():
    data = xr.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("time", "frequency"),
    ).chunk({"time": 3, "frequency": 4})

    result = with_chunks(data, {"antenna": 1})

    assert result is data


def test_with_chunks_rechunks_dataset_when_relevant_dim_exists():
    dataset = xr.Dataset(
        {
            "gain": xr.DataArray(
                np.ones((2, 6), dtype=np.complex128),
                dims=("time", "frequency"),
            ),
            "weight": xr.DataArray(
                np.ones((2, 6), dtype=np.float64),
                dims=("time", "frequency"),
            ),
        }
    ).chunk({"time": 2, "frequency": 6})

    result = with_chunks(dataset, {"frequency": 3})

    assert result is not dataset
    assert result.chunksizes["time"] == (2,)
    assert result.chunksizes["frequency"] == (3, 3)


def test_with_chunks_returns_same_dataset_when_no_relevant_dim():
    dataset = xr.Dataset(
        {
            "gain": xr.DataArray(
                np.ones((2, 6), dtype=np.complex128),
                dims=("time", "frequency"),
            ),
        }
    ).chunk({"time": 2, "frequency": 6})

    result = with_chunks(dataset, {"antenna": 1})

    assert result is dataset


@pytest.mark.skipif(
    not hasattr(xr, "DataTree"),
    reason="xarray.DataTree is not available in this xarray version",
)
def test_with_chunks_returns_same_datatree_for_direct_dim_key():
    dataset = xr.Dataset(
        {
            "gain": xr.DataArray(
                np.ones((2, 6), dtype=np.float64),
                dims=("time", "frequency"),
            )
        }
    ).chunk({"time": 2, "frequency": 6})
    tree = xr.DataTree.from_dict({"node": dataset})
    before_chunks = tree.chunksizes

    result = with_chunks(tree, {"frequency": 3})

    assert result is tree
    assert result.chunksizes["/node"]["time"] == (2,)
    assert result.chunksizes["/node"]["frequency"] == (6,)
    assert result.chunksizes == before_chunks


@pytest.mark.skipif(
    not hasattr(xr, "DataTree"),
    reason="xarray.DataTree is not available in this xarray version",
)
def test_with_chunks_returns_same_datatree_when_no_relevant_dim():
    dataset = xr.Dataset(
        {
            "gain": xr.DataArray(
                np.ones((2, 6), dtype=np.float64),
                dims=("time", "frequency"),
            )
        }
    ).chunk({"time": 2, "frequency": 6})
    tree = xr.DataTree.from_dict({"node": dataset})

    result = with_chunks(tree, {"antenna": 1})

    assert result is tree


def test_should_parse_reference_antenna():
    refant = "LOWBD2_344"
    antennas = ["LOWBD2_344", "LOWBD2_345", "LOWBD2_346", "LOWBD2_347"]
    dims = "id"
    coords = {"id": np.arange(4)}
    ant_names = xr.DataArray(antennas, dims=dims, coords=coords)

    output = parse_antenna(refant, ant_names)

    assert output == 0


def test_should_raise_exception_if_unsuported_type_for_antenna():
    antnames_mock = MagicMock(name="gaintable")
    antnames_mock.size = 5
    with pytest.raises(
        ValueError,
        match=r"Invalid antenna value \[1, 2, 3\]",
    ):
        parse_antenna([1, 2, 3], antnames_mock)


def test_should_raise_error_when_ref_ant_is_invalid():
    refant = "ANTENNA-1"
    antennas = ["LOWBD2_344", "LOWBD2_345", "LOWBD2_346", "LOWBD2_347"]
    dims = "id"
    coords = {"id": np.arange(4)}
    ant_names = xr.DataArray(antennas, dims=dims, coords=coords)

    with pytest.raises(
        ValueError, match="Reference antenna name is not valid"
    ):
        parse_antenna(refant, ant_names)


def test_should_raise_error_when_antenna_index_is_invalid():
    refant = 10
    antnames_mock = MagicMock(name="gaintable")
    antnames_mock.size = 5
    with pytest.raises(ValueError, match="Invalid antenna value 10"):
        parse_antenna(refant, antnames_mock)


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
