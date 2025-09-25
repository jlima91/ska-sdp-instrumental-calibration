import mock
import numpy as np
import pytest
import xarray as xr
from mock import MagicMock

from ska_sdp_instrumental_calibration.processing_tasks.gain_flagging import (
    GainFlagger,
    flag_on_gains,
)


def test_should_flag_gains_for_amplitude():

    soltype = "amplitude"
    mode = "smooth"
    order = 3
    max_rms = 5.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 0.0
    window_noise = 3
    fix_rms_noise = 0.0
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    gains[5] = 0 + 100j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        frequencies,
    )
    updated_weights = flagger_obj.flag_dimension(
        gains, weights, "a1", "X", "Y"
    )
    expected = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    np.testing.assert_allclose(updated_weights, expected)


def test_should_flag_gains_for_phase_with_polyfit():

    soltype = "phase"
    mode = "poly"
    order = 1
    max_rms = 5.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 0.0
    window_noise = 3
    fix_rms_noise = 0.0
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * 0.01j
    gains[5] = 0 + 100j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        frequencies,
    )
    updated_weights = flagger_obj.flag_dimension(
        gains, weights, "a1", "X", "Y"
    )
    expected = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    np.testing.assert_allclose(updated_weights, expected)


def test_should_flag_gains_for_both_phase_and_amplitude():

    soltype = "both"
    mode = "poly"
    order = 2
    max_rms = 4.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 15.0
    window_noise = 1
    fix_rms_noise = 0.0
    frequencies = np.arange(0, 1, 0.1)
    gains = np.zeros(10) + 0j
    gains[5] = -200
    gains[4] = -500j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        frequencies,
    )
    updated_weights = flagger_obj.flag_dimension(
        gains, weights, "a1", "X", "Y"
    )

    expected = [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    np.testing.assert_allclose(updated_weights, expected)


def test_should_throw_exception_if_window_size_is_even():

    soltype = "amplitude"
    mode = "smooth"
    order = 1
    max_rms = 0.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 15.0
    window_noise = 2
    fix_rms_noise = 0.0
    frequencies = np.arange(0, 1, 0.1)
    gains = np.zeros(10) + 0j
    gains[5] = -200
    gains[4] = -500j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        frequencies,
    )

    with pytest.raises(Exception):
        flagger_obj.flag_dimension(gains, weights, "a1", "X", "Y")


@mock.patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "gain_flagging.xr.where"
)
@mock.patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "gain_flagging.xr.apply_ufunc"
)
@mock.patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "gain_flagging.GainFlagger"
)
def test_should_perform_gain_flagging(
    gain_flagger_mock, apply_ufunc_mock, where_mock
):
    soltype = "amplitude"
    mode = "smooth"
    order = 1
    max_rms = 0.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 15.0
    window_noise = 2
    fix_rms_noise = 0.0

    nstations = 2
    nfreq = 5

    gain_data = np.ones((1, nstations, nfreq, 2, 2)) + 1j * np.ones(
        (1, nstations, nfreq, 2, 2)
    )
    antenna_coords = [f"{i}" for i in range(nstations)]
    freq_coords = np.linspace(1e8, 2e8, nfreq)

    dims = ("time", "antenna", "frequency", "receptor1", "receptor2")
    coords = {
        "time": [0],
        "antenna": antenna_coords,
        "frequency": freq_coords,
        "receptor1": ["X", "Y"],
        "receptor2": ["X", "Y"],
    }

    weights = MagicMock(name="weights")

    gaintable_mock = MagicMock(name="gaintable")
    gaintable_mock.chunk.return_value = gaintable_mock
    gaintable_mock.assign.return_value = gaintable_mock
    gaintable_mock.receptor1 = xr.DataArray(
        ["X", "Y"], dims="id", coords={"id": np.arange(2)}
    )
    gaintable_mock.receptor2 = xr.DataArray(
        ["X", "Y"], dims="id", coords={"id": np.arange(2)}
    )
    gaintable_mock.gain = xr.DataArray(gain_data, coords=coords, dims=dims)
    gaintable_mock.weights = weights
    gaintable_mock.weight.copy.return_value = weights

    apply_ufunc_return_values = [
        xr.DataArray(
            np.array([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]]),
        ),
        xr.DataArray(
            np.array([[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]]),
        ),
    ]

    gain_flagger_mock.return_value = gain_flagger_mock
    apply_ufunc_mock.side_effect = apply_ufunc_return_values

    where_mock.return_value = "NEW_GAIN"

    gaintable = flag_on_gains(
        gaintable_mock,
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        skip_cross_pol=True,
        apply_flag=True,
    )

    where_mock.assert_called_once_with(weights, 0.0, gaintable_mock["gain"])

    assert apply_ufunc_mock.call_count == 2

    gaintable.assign.assert_called_once_with(
        {"gain": "NEW_GAIN", "weight": weights}
    )

    expected_weights = np.array(
        [
            [
                [
                    [[1, 1], [1, 1]],
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                ],
                [
                    [[1, 1], [1, 1]],
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                ],
            ]
        ]
    )

    assert np.all(weights.data == expected_weights)
    gaintable.chunk.assert_has_calls(
        [mock.call({"frequency": -1}), mock.call(gaintable.chunks)]
    )


@mock.patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "gain_flagging.xr.apply_ufunc"
)
@mock.patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "gain_flagging.GainFlagger"
)
def test_should_perform_gain_flagging_without_apply(
    gain_flagger_mock, apply_ufunc_mock
):
    soltype = "amplitude"
    mode = "smooth"
    order = 1
    max_rms = 0.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 15.0
    window_noise = 2
    fix_rms_noise = 0.0

    nstations = 2
    nfreq = 5

    gain_data = np.ones((1, nstations, nfreq, 2, 2)) + 1j * np.ones(
        (1, nstations, nfreq, 2, 2)
    )
    antenna_coords = [f"{i}" for i in range(nstations)]
    freq_coords = np.linspace(1e8, 2e8, nfreq)

    dims = ("time", "antenna", "frequency", "receptor1", "receptor2")
    coords = {
        "time": [0],
        "antenna": antenna_coords,
        "frequency": freq_coords,
        "receptor1": ["X", "Y"],
        "receptor2": ["X", "Y"],
    }

    weights = MagicMock(name="weights")

    gaintable_mock = MagicMock(name="gaintable")
    gaintable_mock.chunk.return_value = gaintable_mock
    gaintable_mock.assign.return_value = gaintable_mock
    gaintable_mock.receptor1 = xr.DataArray(
        ["X", "Y"], dims="id", coords={"id": np.arange(2)}
    )
    gaintable_mock.receptor2 = xr.DataArray(
        ["X", "Y"], dims="id", coords={"id": np.arange(2)}
    )
    gaintable_mock.gain = xr.DataArray(gain_data, coords=coords, dims=dims)
    gaintable_mock.weights = weights
    gaintable_mock.weight.copy.return_value = weights

    apply_ufunc_return_values = [
        xr.DataArray(np.array([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]])),
        xr.DataArray(np.array([[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]])),
        xr.DataArray(np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]])),
        xr.DataArray(np.array([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]])),
    ]

    gain_flagger_mock.return_value = gain_flagger_mock
    apply_ufunc_mock.side_effect = apply_ufunc_return_values

    gaintable = flag_on_gains(
        gaintable_mock,
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        skip_cross_pol=False,
        apply_flag=False,
    )

    assert apply_ufunc_mock.call_count == 4

    gaintable.assign.assert_called_once_with({"weight": weights})

    expected_weights = np.array(
        [
            [
                [
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                ],
                [
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[1, 1], [1, 1]],
                ],
            ]
        ]
    )

    assert np.all(weights.data == expected_weights)
    gaintable.chunk.assert_has_calls(
        [mock.call({"frequency": -1}), mock.call(gaintable.chunks)]
    )
