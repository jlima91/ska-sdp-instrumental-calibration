import numpy as np
import pytest
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_visibility,
    divide_bandpass_by_ref_ant_preserve_phase,
    reset_gaintable,
)


def test_should_raise_exception_for_jonestype(generate_vis):
    vis, jones = generate_vis
    with pytest.raises(ValueError, match="Unknown Jones type X"):
        create_gaintable_from_visibility(vis, jones_type="X")


def test_should_generate_gaintable_with_defaults(generate_vis):
    vis, jones = generate_vis
    gaintable = create_gaintable_from_visibility(vis)

    expected_frequency = np.mean(vis.frequency.data, keepdims=True)
    gain_shape = (len(vis.time.data), vis.visibility_acc.nants, 1, 2, 2)
    residual_shape = (len(vis.time.data), 1, 2, 2)

    np.testing.assert_allclose(gaintable.frequency.data, expected_frequency)
    assert gaintable.gain.shape == gain_shape
    assert gaintable.weight.shape == gain_shape
    assert gaintable.residual.shape == residual_shape
    assert gaintable.soln_interval_slices == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
    ]


def test_should_generate_gaintable_for_jonetypes_B(generate_vis):
    vis, jones = generate_vis
    vis = vis.chunk(frequency=1)
    gaintable = create_gaintable_from_visibility(vis, jones_type="B")

    gain_shape = (
        len(vis.time.data),
        vis.visibility_acc.nants,
        len(vis.frequency.data),
        2,
        2,
    )

    np.testing.assert_allclose(gaintable.frequency.data, vis.frequency.data)
    assert gaintable.gain.shape == gain_shape
    assert gaintable.weight.shape == gain_shape
    assert gaintable.soln_interval_slices == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
    ]


def test_should_skip_default_chunk(generate_vis):
    vis, jones = generate_vis
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", skip_default_chunk=True
    )

    gain_shape = (
        len(vis.time.data),
        vis.visibility_acc.nants,
        len(vis.frequency.data),
        2,
        2,
    )

    np.testing.assert_allclose(gaintable.frequency.data, vis.frequency.data)
    assert gaintable.gain.shape == gain_shape
    assert gaintable.weight.shape == gain_shape
    assert gaintable.soln_interval_slices == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
    ]


@patch("ska_sdp_instrumental_calibration.data_managers.gaintable.da")
def test_should_reset_gaintable(da_mock, generate_vis):
    vis, jones = generate_vis
    gaintable = MagicMock(name="gaintable")
    r_gaintable = reset_gaintable(gaintable)
    da_mock.eye.asserrt_called_once_with(
        gaintable.gain.shape[-1], dtype=gaintable.gain.dtype
    )

    da_mock.broadcast_to.assert_called_once_with(
        da_mock.eye.return_value, gaintable.gain.shape
    )

    da_mock.ones.assert_called_once_with(
        gaintable.weight.shape, dtype=gaintable.weight.dtype
    )

    da_mock.zeros.assert_called_once_with(
        gaintable.residual.shape, dtype=gaintable.residual.dtype
    )

    gaintable.copy.assert_called_once_with(deep=True)
    assert r_gaintable == gaintable.copy.return_value
    assert r_gaintable.gain.data == da_mock.broadcast_to.return_value
    assert r_gaintable.weight.data == da_mock.ones.return_value
    assert r_gaintable.residual.data == da_mock.zeros.return_value


def test_should_divide_bandpass_by_ref_ant_and_preserve_phase(generate_vis):
    vis, _ = generate_vis
    vis = vis.chunk(frequency=1)
    gaintable = create_gaintable_from_visibility(vis, jones_type="B")

    actual_gaintable = divide_bandpass_by_ref_ant_preserve_phase(gaintable, 0)
    complex_gains = actual_gaintable.gain.data

    x_angle = np.angle(complex_gains[:, 0, :, 0, 0])
    y_angle = np.angle(complex_gains[:, 0, :, 1, 1])

    actual_amp = np.abs(complex_gains)
    expected_amp = np.abs(gaintable.gain.data)

    assert np.allclose(x_angle[np.isfinite(x_angle)], 0)
    assert np.allclose(y_angle[np.isfinite(y_angle)], 0)
    assert np.allclose(actual_amp, expected_amp)
