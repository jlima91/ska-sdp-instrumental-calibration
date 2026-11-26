import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors.vis_filter import (
    BaselineFilter,
    VisibilityFilter,
)


def _flags(nbaselines):
    return xr.DataArray(
        np.zeros((2, nbaselines, 5, 4), dtype=bool),
        dims=["time", "baseline_id", "frequency", "polarization"],
        coords={
            "time": np.arange(2),
            "baseline_id": np.arange(nbaselines),
            "frequency": np.arange(5),
            "polarization": ["XX", "XY", "YX", "YY"],
        },
    )


def _ms_xdt(baselines):
    """MSv4 node with antennas ANT1, ANT2 and the given baselines."""
    vis = xr.Dataset(
        {"FLAG": _flags(len(baselines))},
        coords={
            "baseline_antenna1_name": (
                "baseline_id",
                [a1 for a1, _ in baselines],
            ),
            "baseline_antenna2_name": (
                "baseline_id",
                [a2 for _, a2 in baselines],
            ),
        },
        attrs={"data_groups": {"base": {"flag": "FLAG"}}},
    )
    antenna_xds = xr.Dataset(coords={"antenna_name": ["ANT1", "ANT2"]})
    return xr.DataTree.from_dict({"/": vis, "/antenna_xds": antenna_xds})


@pytest.fixture
def vis():
    yield _ms_xdt([("ANT1", "ANT1"), ("ANT1", "ANT2"), ("ANT2", "ANT2")])


def test_no_baselines_to_ignore_returns_flags_unchanged(vis):
    result = BaselineFilter._filter("", vis)
    assert (result == vis.ds.FLAG).all()


def test_ingnore_single_baseline(vis):
    result = BaselineFilter._filter("ANT1&ANT2", vis)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()

    result = BaselineFilter._filter("0&1", vis)

    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()


def test_filter_baselines(vis):
    filtered_vis = VisibilityFilter.filter(
        {"exclude_baselines": "ANT1&ANT2"}, vis
    )

    result = filtered_vis.ds["flag"]

    # Known issue kept from the pre-migration implementation: the filter
    # writes a separate "flag" variable, so the data group flags used by
    # the solvers are unchanged
    assert filtered_vis.ds.FLAG.equals(vis.ds.FLAG)
    assert not vis.ds.FLAG.any()
    assert not result[:, 0, :, :].all()
    assert result[:, 1, :, :].all()
    assert not result[:, 2, :, :].all()


def test_filter_multiple_baselines(vis):
    result = BaselineFilter._filter("0&1,1&1", vis)

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


def test_should_match_baselines_in_stored_antenna_order():
    # Kept from the pre-migration implementation: an expression matches the
    # antenna order of the baseline only
    vis = _ms_xdt([("ANT2", "ANT1")])

    assert not BaselineFilter._filter("ANT1&ANT2", vis).any()
    assert BaselineFilter._filter("ANT2&ANT1", vis).all()
