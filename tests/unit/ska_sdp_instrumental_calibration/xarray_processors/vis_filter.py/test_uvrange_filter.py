import numpy as np
import pytest
import xarray as xr
from astropy import constants as const
from mock import ANY, MagicMock, call, patch

from ska_sdp_instrumental_calibration.xarray_processors.vis_filter import (
    UVRangeFilter,
    VisibilityFilter,
)


@pytest.mark.parametrize(
    "uvrange, expected_uvrange",
    [
        (
            "0~10klambda,100~500",
            [
                {
                    "uv_min": 0,
                    "uv_max": 10,
                    "unit": "kl",
                    "negate": False,
                    "min_inclusive": True,
                    "max_inclusive": True,
                },
                {
                    "uv_min": 100,
                    "uv_max": 500,
                    "unit": "m",
                    "negate": False,
                    "min_inclusive": True,
                    "max_inclusive": True,
                },
            ],
        ),
        (
            ">500m,<10kl",
            [
                {
                    "uv_min": 500,
                    "uv_max": np.inf,
                    "unit": "m",
                    "negate": False,
                    "min_inclusive": False,
                    "max_inclusive": True,
                },
                {
                    "uv_min": -np.inf,
                    "uv_max": 10,
                    "unit": "kl",
                    "negate": False,
                    "min_inclusive": True,
                    "max_inclusive": False,
                },
            ],
        ),
        (
            "!>500m",
            [
                {
                    "uv_min": 500,
                    "uv_max": np.inf,
                    "unit": "m",
                    "negate": True,
                    "min_inclusive": False,
                    "max_inclusive": True,
                },
            ],
        ),
        (
            "10~100kilometer",
            [
                {
                    "uv_min": 10,
                    "uv_max": 100,
                    "unit": "km",
                    "negate": False,
                    "min_inclusive": True,
                    "max_inclusive": True,
                },
            ],
        ),
    ],
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".vis_filter.uvrange_filter.UVRange"
)
def test_should_create_uvranges_for_the_given_uvrange_strings(
    uvrange_mock, uvrange, expected_uvrange
):
    UVRangeFilter(uvrange)

    uvrange_mock.assert_has_calls(
        [call(**uvrange_args) for uvrange_args in expected_uvrange]
    )


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".vis_filter.uvrange_filter.UVRange"
)
def test_should_not_uvdist_kl(uvrange_mock):
    uvrange_mock.return_value = uvrange_mock

    uv_range_filter = UVRangeFilter("<10km")

    u = xr.DataArray(
        np.zeros((2, 4)),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    v = xr.DataArray(
        np.zeros((2, 4)),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    freq = xr.DataArray(np.arange(5), dims=["frequency"])

    flags = xr.DataArray(
        np.zeros((2, 4, 5, 4), dtype=bool),
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(4),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )

    uvrange_mock.predicate.return_value = u > 4

    uv_range_filter(u, v, flags, freq)

    uvrange_mock.predicate.assert_called_once_with(ANY, None)

    call_arguments = uvrange_mock.predicate.call_args[0]
    assert np.array_equal(call_arguments[0], np.hypot(u, v))
    assert call_arguments[1] is None


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".vis_filter.uvrange_filter.UVRange"
)
def test_should_precompute_uvdist_kl(uvrange_mock):
    uvrange_mock.return_value = uvrange_mock

    uv_range_filter = UVRangeFilter("<10kl")

    u = xr.DataArray(
        np.zeros((2, 4)),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    v = xr.DataArray(
        np.zeros((2, 4)),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    freq = xr.DataArray(np.arange(5), dims=["frequency"])

    flags = xr.DataArray(
        np.zeros((2, 4, 5, 4), dtype=bool),
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(4),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )

    uvrange_mock.predicate.return_value = u > 4

    uv_range_filter(u, v, flags, freq)

    uvrange_mock.predicate.assert_called_once_with(ANY, ANY)

    call_arguments = uvrange_mock.predicate.call_args[0]
    assert np.array_equal(call_arguments[0], np.hypot(u, v))

    _lambda = const.c.value / freq  # pylint: disable=no-member
    uvdist_kl = (np.hypot(u, v) / _lambda) / 1000.0
    assert np.array_equal(call_arguments[1], uvdist_kl)


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".vis_filter.uvrange_filter.UVRange"
)
def test_should_update_flags(uvrange_mock):
    uvrange_mock.return_value = uvrange_mock
    vis = MagicMock(name="vis")

    vis.visibility_acc.u = xr.DataArray(
        np.array([[3, 4, 5, 6], [7, 8, 9, 10]]),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    vis.visibility_acc.v = xr.DataArray(
        np.zeros((2, 4)),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    vis.frequency = xr.DataArray(np.arange(5), dims=["frequency"])

    vis.flags = xr.DataArray(
        np.zeros((2, 4, 5, 4), dtype=bool),
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(4),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )

    uvrange_mock.predicate.return_value = vis.visibility_acc.u > 6
    filtered_vis = VisibilityFilter.filter({"uvdist": "<10kl"}, vis)

    expected = np.zeros((2, 4, 5, 4), dtype=bool)
    expected[0, ...] = True

    expected_flags = xr.DataArray(
        expected,
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(4),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )

    vis.assign.assert_called_once_with({"flag": ANY})

    called_args, _ = vis.assign.call_args

    actual = called_args[0]["flag"]

    assert filtered_vis == vis.assign.return_value
    assert actual.equals(expected_flags)


def test_should_raise_value_error_for_invalid_uvrange_string():
    with pytest.raises(
        ValueError, match="Could not parse uvrange string: 'invalid'"
    ):
        UVRangeFilter("invalid")


def test_should_raise_value_error_for_invalid_unit():
    with pytest.raises(ValueError, match="Unknown unit: xyz"):
        UVRangeFilter(">500xyz")


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".vis_filter.uvrange_filter.np.hypot"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".vis_filter.uvrange_filter.UVRange"
)
def test_should_raise_freq_value_error_for_klambda_selection(_, __):
    with pytest.raises(
        ValueError, match="Frequency required for 'klambda' selection."
    ):
        uv_filter = UVRangeFilter("0~10klambda")
        uv_filter("u", "v", "flags")


def test_should_update_not_flags():

    uv_range_filter = UVRangeFilter(None)

    u = xr.DataArray(
        np.array([[3, 4, 5, 6], [7, 8, 9, 10]]),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    v = xr.DataArray(
        np.zeros((2, 4)),
        dims=["time", "baseline"],
        coords={"time": np.arange(2), "baseline": np.arange(4)},
    )

    flags = xr.DataArray(
        np.zeros((2, 4, 5, 4), dtype=bool),
        dims=["time", "baseline", "frequency", "pol"],
        coords={
            "time": np.arange(2),
            "baseline": np.arange(4),
            "frequency": np.arange(5),
            "pol": np.arange(4),
        },
    )

    actual = uv_range_filter(u, v, flags)

    assert actual.equals(flags)
