import pytest
from mock import MagicMock

from ska_sdp_instrumental_calibration.xarray_processors.vis_filter import (
    BaselineFilter,
    UVRangeFilter,
    VisibilityFilter,
    vis_filter,
)


def test_should_have_registered_the_filters():
    assert VisibilityFilter._data_filters["uvdist"] == UVRangeFilter
    assert VisibilityFilter._data_filters["baselines"] == BaselineFilter


def test_should_raise_exception_for_unsupported_filter():
    with pytest.raises(
        ValueError, match="Strategy for invalid filter not known"
    ):
        VisibilityFilter.filter({"invalid": "FILTER"}, "vis")


def test_should_not_filter_if_filter_expr_is_none():
    vis = MagicMock(name="VIS")

    expected_vis = VisibilityFilter.filter(None, vis)
    assert expected_vis == vis

    expected_vis = VisibilityFilter.filter({"uvdist": None}, vis)
    assert expected_vis == vis

    vis.assign.assert_not_called()


def test_should_raise_exception_for_not_implemented_filter_function():
    with pytest.raises(NotImplementedError):
        vis_filter.AbstractVisibilityFilter.filter("filter", "vis")
