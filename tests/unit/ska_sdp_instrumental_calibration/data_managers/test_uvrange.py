import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.uv_range import UVRange


def test_should_return_predicate_condition_by_range():
    params = {
        "uv_min": 10,
        "uv_max": 100,
        "unit": "m",
        "negate": False,
        "min_inclusive": True,
        "max_inclusive": True,
    }
    uvrange = UVRange(**params)
    uvdist = np.array([[1, 10, 50, 110], [5, 30, 100, 120]])

    actual_cond = uvrange.predicate(uvdist, None)
    expected_cond = np.array(
        [[False, True, True, False], [False, True, True, False]]
    )

    np.testing.assert_array_equal(actual_cond, expected_cond)


def test_should_raise_value_error_for_invalid_range():
    params = {
        "uv_min": 100,
        "uv_max": 10,
        "unit": "m",
        "negate": False,
        "min_inclusive": True,
        "max_inclusive": True,
    }

    with pytest.raises(
        ValueError, match=r"Invalid range: min > max \(100 > 10\)"
    ):
        UVRange(**params)


def test_should_raise_unknown_unit_error():
    params = {
        "uv_min": 10,
        "uv_max": 100,
        "unit": "unknown",
        "negate": False,
        "min_inclusive": True,
        "max_inclusive": True,
    }
    uvrange = UVRange(**params)
    uvdist = np.array([[1, 20, 50, 110], [5, 30, 80, 120]])

    with pytest.raises(ValueError, match="Unknown unit: unknown"):
        uvrange.predicate(uvdist, None)


def test_should_negate_condition():
    params = {
        "uv_min": 10,
        "uv_max": 100,
        "unit": "kl",
        "negate": True,
        "min_inclusive": True,
        "max_inclusive": True,
    }
    uvrange = UVRange(**params)
    uvdist = np.array([[1, 20, 50, 110], [5, 30, 80, 120]])

    actual_cond = uvrange.predicate(uvdist, uvdist)
    expected_cond = np.array(
        [[True, False, False, True], [True, False, False, True]]
    )

    np.testing.assert_array_equal(actual_cond, expected_cond)


def test_should_not_include_minimum_if_min_max_inclusive_false():
    params = {
        "uv_min": 10,
        "uv_max": 100,
        "unit": "km",
        "negate": False,
        "min_inclusive": False,
        "max_inclusive": False,
    }
    uvrange = UVRange(**params)
    uvdist = np.array(
        [[10000, 20000, 50000, 100000], [5000, 10000, 80000, 120000]]
    )

    actual_cond = uvrange.predicate(uvdist, None)
    expected_cond = np.array(
        [[False, True, True, False], [False, False, True, False]]
    )

    np.testing.assert_array_equal(actual_cond, expected_cond)
