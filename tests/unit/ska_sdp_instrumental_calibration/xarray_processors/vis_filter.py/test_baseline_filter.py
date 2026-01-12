import numpy as np
import pandas
import pytest
import xarray as xr
from mock import ANY, MagicMock

from ska_sdp_instrumental_calibration.xarray_processors.vis_filter import (
    BaselineFilter,
    VisibilityFilter,
)


@pytest.fixture
def vis():
    vis = MagicMock(name="vis")

    vis.flags = xr.DataArray(
        np.zeros((2, 3, 5, 4), dtype=bool),
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(3),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )

    _baselines = pandas.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 1)], names=["ANT1", "ANT2"]
    )
    vis.baselines = xr.DataArray(
        _baselines, dims=["baseline"], coords={"baseline": np.arange(3)}
    )
    vis.configuration.names = xr.DataArray(
        ["ANT1", "ANT2"], coords={"id": [0, 1]}, dims=["id"]
    )

    yield vis


def test_no_baselines_to_ignore_returns_flags_unchanged(vis):
    result = BaselineFilter._filter("", vis)
    assert (result == vis.flags).all()


def test_ingnore_single_baseline(vis):
    result = BaselineFilter._filter("!ANT1&ANT2", vis)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()

    result = BaselineFilter._filter("!0&1", vis)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()


def test_filter_baselines(vis):
    filtered_vis = VisibilityFilter.filter(
        {"exclude_baselines": "!ANT1&ANT2"}, vis
    )

    vis.assign.assert_called_once_with({"flag": ANY})

    called_args, _ = vis.assign.call_args

    result = called_args[0]["flag"]

    assert filtered_vis == vis.assign.return_value
    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()


def test_filter_multiple_baselines(vis):
    result = BaselineFilter._filter("!ANT1&ANT2,ANT1&ANT1", vis)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert result[:, 2, :, :].all()

    result = BaselineFilter._filter("ANT1~ANT2&ANT2,ANT1&ANT1", vis)

    assert result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert result[:, 2, :, :].all()


def test_invalid_baseline_string_raises(vis):
    with pytest.raises(
        ValueError, match="Invalid baseline expression: ANT1-ANT2"
    ):
        BaselineFilter._filter("ANT1-ANT2", vis)
