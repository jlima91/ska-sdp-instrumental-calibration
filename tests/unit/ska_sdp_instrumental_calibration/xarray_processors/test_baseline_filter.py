import numpy as np
import pandas
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors import baseline_filter

BaselineFilter = baseline_filter.BaselineFilter


@pytest.fixture
def station_names():
    return xr.DataArray(["ANT1", "ANT2"], coords={"id": [0, 1]}, dims=["id"])


@pytest.fixture
def station_counts():
    return 2


@pytest.fixture
def baselines():
    _baselines = pandas.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 1)], names=["ANT1", "ANT2"]
    )
    return xr.DataArray(
        _baselines, dims=["baseline"], coords={"baseline": np.arange(3)}
    )


@pytest.fixture
def flags():
    return xr.DataArray(
        np.zeros((2, 3, 5, 4), dtype=bool),
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(3),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )


def test_no_baselines_to_ignore_returns_flags_unchanged(
    station_names, station_counts, baselines, flags
):
    bf = BaselineFilter("", station_names, station_counts)
    result = bf(baselines, flags)
    assert (result == flags).all()


def test_ignore_single_baseline(
    baselines, flags, station_names, station_counts
):
    bf = BaselineFilter("ANT1&ANT2", station_names, station_counts)

    result = bf(baselines, flags)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()

    bf = BaselineFilter("0&1", station_names, station_counts)

    result = bf(baselines, flags)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()


def test_ignore_multiple_baselines(
    baselines, flags, station_names, station_counts
):
    bf = BaselineFilter("ANT1&ANT2,ANT1&ANT1", station_names, station_counts)
    result = bf(baselines, flags)

    assert result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()


def test_invalid_baseline_string_raises(station_names, station_counts):
    with pytest.raises(
        ValueError, match="Could not parse baselines from 'ANT1-ANT2'"
    ):
        BaselineFilter("ANT1-ANT2", station_names, station_counts)
