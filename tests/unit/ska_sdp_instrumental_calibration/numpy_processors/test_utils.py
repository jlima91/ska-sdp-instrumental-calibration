import dask.array as da
import numpy as np
import pytest

from ska_sdp_instrumental_calibration.numpy_processors._utils import stack_2x2


def test_stack_2x2_all_inputs():
    xx = np.array([[1, 2], [3, 4]])
    xy = np.array([[5, 6], [7, 8]])
    yx = np.array([[9, 10], [11, 12]])
    yy = np.array([[13, 14], [15, 16]])

    result = stack_2x2(xx, xy, yx, yy)

    assert result.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(result[..., 0, 0], xx)
    np.testing.assert_array_equal(result[..., 0, 1], xy)
    np.testing.assert_array_equal(result[..., 1, 0], yx)
    np.testing.assert_array_equal(result[..., 1, 1], yy)


def test_stack_2x2_missing_off_diagonal():
    xx = np.ones((3, 4))
    yy = np.full((3, 4), 2.0)

    result = stack_2x2(xx=xx, yy=yy)

    assert result.shape == (3, 4, 2, 2)
    np.testing.assert_array_equal(result[..., 0, 0], xx)
    np.testing.assert_array_equal(result[..., 1, 1], yy)
    np.testing.assert_array_equal(result[..., 0, 1], 0.0)
    np.testing.assert_array_equal(result[..., 1, 0], 0.0)


def test_stack_2x2_missing_diagonal():
    xy = np.ones((2, 5))
    yx = np.full((2, 5), 3.0)

    result = stack_2x2(xy=xy, yx=yx)

    assert result.shape == (2, 5, 2, 2)
    np.testing.assert_array_equal(result[..., 0, 1], xy)
    np.testing.assert_array_equal(result[..., 1, 0], yx)
    np.testing.assert_array_equal(result[..., 0, 0], 0.0)
    np.testing.assert_array_equal(result[..., 1, 1], 0.0)


def test_stack_2x2_only_xx():
    xx = np.arange(6, dtype=float).reshape(2, 3)

    result = stack_2x2(xx=xx)

    assert result.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(result[..., 0, 0], xx)
    np.testing.assert_array_equal(result[..., 0, 1], 0.0)
    np.testing.assert_array_equal(result[..., 1, 0], 0.0)
    np.testing.assert_array_equal(result[..., 1, 1], 0.0)


def test_stack_2x2_raises_when_all_none():
    with pytest.raises(
        ValueError, match="At least one input array must be provided"
    ):
        stack_2x2()


def test_stack_2x2_dask_arrays():
    xx = da.from_array(np.ones((3, 4)), chunks=2)
    yy = da.from_array(np.full((3, 4), 2.0), chunks=2)

    result = stack_2x2(xx=xx, yy=yy)

    assert isinstance(result, da.Array)
    computed = result.compute()
    assert computed.shape == (3, 4, 2, 2)
    np.testing.assert_array_equal(computed[..., 0, 0], 1.0)
    np.testing.assert_array_equal(computed[..., 1, 1], 2.0)
    np.testing.assert_array_equal(computed[..., 0, 1], 0.0)
    np.testing.assert_array_equal(computed[..., 1, 0], 0.0)


def test_stack_2x2_higher_rank():
    xx = np.ones((2, 3, 4))
    xy = np.full((2, 3, 4), 2.0)
    yx = np.full((2, 3, 4), 3.0)
    yy = np.full((2, 3, 4), 4.0)

    result = stack_2x2(xx, xy, yx, yy)

    assert result.shape == (2, 3, 4, 2, 2)
    np.testing.assert_array_equal(result[..., 0, 0], 1.0)
    np.testing.assert_array_equal(result[..., 0, 1], 2.0)
    np.testing.assert_array_equal(result[..., 1, 0], 3.0)
    np.testing.assert_array_equal(result[..., 1, 1], 4.0)
