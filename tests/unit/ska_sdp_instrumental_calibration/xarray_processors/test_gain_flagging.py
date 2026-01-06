import mock
import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.xarray_processors.gain_flagging import (
    GainFlagger,
    flag_on_gains,
)


def test_should_flag_gains_for_amplitude():

    soltype = "amplitude"
    mode = "smooth"
    order = 3
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    gains[5] = 0 + 100j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
        normalize_gains=False,
    )
    updated_weights = flagger_obj.flag_dimension(
        weights, gains, "a1", "X", "Y"
    )

    amp_fit = [
        2.21575891,
        2.19544984,
        2.16333077,
        2.14009346,
        2.14009346,
        2.12602916,
        2.14009346,
        2.14009346,
        2.16333077,
        2.1793903,
    ]
    phase_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    real_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    imag_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    flagged_weights = [
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
    expected = flagged_weights, amp_fit, phase_fit, real_fit, imag_fit

    np.testing.assert_allclose(updated_weights, expected)


def test_should_flag_gains_for_phase_with_polyfit():

    soltype = "phase"
    mode = "poly"
    order = 1
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * 0.01j
    gains[5] = 0 + 100j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
        normalize_gains=False,
    )
    updated_weights = flagger_obj.flag_dimension(
        weights, gains, "a1", "X", "Y"
    )
    amp_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    real_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    imag_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    phase_fit = [
        0.11423973,
        0.12375971,
        0.13327969,
        0.14279967,
        0.15231964,
        0.16183962,
        0.1713596,
        0.18087958,
        0.19039955,
        0.19991953,
    ]
    flagged_weights = [
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
    expected = flagged_weights, amp_fit, phase_fit, real_fit, imag_fit

    np.testing.assert_allclose(updated_weights, expected)


def test_should_flag_gains_for_both_phase_and_amplitude():

    soltype = "amp-phase"
    mode = "poly"
    order = 2
    n_sigma = None
    max_ncycles = 1
    n_sigma_rolling = 5.0
    window_size = 3
    frequencies = np.arange(0, 1, 0.1)
    gains = np.zeros(10) + 0j
    gains[5] = -200
    gains[4] = -500j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
        normalize_gains=True,
    )
    (
        updated_weights,
        updated_amp_fit,
        updated_phase_fit,
        updated_real_fit,
        updated_imag_fit,
    ) = flagger_obj.flag_dimension(weights, gains, "a1", "X", "Y")
    amp_fit = [
        -0.07013,
        0.048485,
        0.136797,
        0.194805,
        0.222511,
        0.219913,
        0.187013,
        0.12381,
        0.030303,
        -0.093506,
    ]
    phase_fit = [
        -2.57039399e-01,
        -3.80799110e-02,
        1.33279688e-01,
        2.57039399e-01,
        3.33199221e-01,
        3.61759154e-01,
        3.42719199e-01,
        2.76079354e-01,
        1.61839622e-01,
        -1.55431223e-15,
    ]
    real_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    imag_fit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    flagged_weights = [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]

    np.testing.assert_allclose(updated_weights, flagged_weights)
    np.testing.assert_allclose(updated_amp_fit, amp_fit, rtol=1e-5)
    np.testing.assert_allclose(updated_phase_fit, phase_fit)
    np.testing.assert_allclose(updated_real_fit, real_fit)
    np.testing.assert_allclose(updated_imag_fit, imag_fit)


def test_should_flag_gains_for_real_imag():

    soltype = "real-imag"
    mode = "smooth"
    order = 3
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    gains[5] = 100 + 100j  # Outlier in both real and imaginary
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
        normalize_gains=False,
    )
    flagged_weights, amp_fit, phase_fit, real_fit, imag_fit = (
        flagger_obj.flag_dimension(weights, gains, "a1", "X", "Y")
    )

    expected_flagged_weights = np.array(
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    )
    expected_amp_fit = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    expected_phase_fit = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    expected_real_fit = np.array(
        [
            1.0500,
            1.1000,
            1.2000,
            1.3000,
            1.4000,
            1.6000,
            1.7000,
            1.7000,
            1.8000,
            1.8500,
        ]
    )
    expected_imag_fit = np.array(
        [
            1.9000,
            1.8500,
            1.8000,
            1.7000,
            1.6500,
            1.6000,
            1.3000,
            1.2500,
            1.2500,
            1.2000,
        ]
    )

    np.testing.assert_allclose(flagged_weights, expected_flagged_weights)
    np.testing.assert_allclose(amp_fit, expected_amp_fit)
    np.testing.assert_allclose(phase_fit, expected_phase_fit)
    np.testing.assert_allclose(real_fit, expected_real_fit)
    np.testing.assert_allclose(imag_fit, expected_imag_fit)


def test_should_throw_exception_if_window_size_is_even():

    soltype = "amplitude"
    mode = "smooth"
    order = 1
    n_sigma = 0.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 2
    frequencies = np.arange(0, 1, 0.1)
    gains = np.zeros(10) + 0j
    gains[5] = -200
    gains[4] = -500j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
        normalize_gains=False,
    )

    with pytest.raises(Exception):
        flagger_obj.flag_dimension(gains, weights, "a1", "X", "Y")


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "gain_flagging.xr.where"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "gain_flagging.xr.apply_ufunc"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "gain_flagging.GainFlagger"
)
def test_should_perform_gain_flagging(
    gain_flagger_mock, apply_ufunc_mock, where_mock
):
    soltype = "amplitude"
    mode = "smooth"
    order = 1
    n_sigma = 0.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 2

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
    gaintable_mock.weight.copy.return_value = xr.DataArray(
        np.ones((1, 2, 5, 2, 2))
    )
    gaintable_mock.weight.data = np.ones((1, nstations, nfreq, 2, 2))

    dims = ("antenna", "frequency")
    coords = {
        "antenna": [f"{i}" for i in range(nstations)],
        "frequency": freq_coords,
    }

    weight_flag_1 = xr.DataArray(
        np.array([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]]), dims=dims, coords=coords
    )
    amp_fit_1 = xr.DataArray(
        np.array([[0.1, 0.1, 0.1, 0.0, 0.1], [0.1, 0.1, 0.1, 0.0, 0.1]]),
        dims=dims,
        coords=coords,
    )
    phase_fit_1 = xr.DataArray(
        np.array([[0.2, 0.2, 0.2, 0.0, 0.2], [0.2, 0.2, 0.2, 0.0, 0.2]]),
        dims=dims,
        coords=coords,
    )
    real_fit_1 = xr.DataArray(
        np.array(
            [[0.15, 0.15, 0.15, 0.0, 0.15], [0.15, 0.15, 0.15, 0.0, 0.15]]
        ),
        dims=dims,
        coords=coords,
    )
    imag_fit_1 = xr.DataArray(
        np.array(
            [[0.25, 0.25, 0.25, 0.0, 0.25], [0.25, 0.25, 0.25, 0.0, 0.25]]
        ),
        dims=dims,
        coords=coords,
    )

    weight_flag_2 = xr.DataArray(
        np.array([[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]]), dims=dims, coords=coords
    )
    amp_fit_2 = xr.DataArray(
        np.array([[0.3, 0.0, 0.3, 0.3, 0.3], [0.3, 0.0, 0.3, 0.3, 0.3]]),
        dims=dims,
        coords=coords,
    )
    phase_fit_2 = xr.DataArray(
        np.array([[0.4, 0.0, 0.4, 0.4, 0.4], [0.4, 0.0, 0.4, 0.4, 0.4]]),
        dims=dims,
        coords=coords,
    )
    real_fit_2 = xr.DataArray(
        np.array(
            [[0.35, 0.0, 0.35, 0.35, 0.35], [0.35, 0.0, 0.35, 0.35, 0.35]]
        ),
        dims=dims,
        coords=coords,
    )
    imag_fit_2 = xr.DataArray(
        np.array(
            [[0.45, 0.0, 0.45, 0.45, 0.45], [0.45, 0.0, 0.45, 0.45, 0.45]]
        ),
        dims=dims,
        coords=coords,
    )

    apply_ufunc_mock.side_effect = [
        (weight_flag_1, amp_fit_1, phase_fit_1, real_fit_1, imag_fit_1),
        (weight_flag_2, amp_fit_2, phase_fit_2, real_fit_2, imag_fit_2),
    ]

    gain_flagger_mock.return_value = gain_flagger_mock

    where_mock.return_value = "NEW_GAIN"

    result_gaintable, fits = flag_on_gains(
        gaintable_mock,
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains=False,
        skip_cross_pol=True,
        apply_flag=True,
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

    expected_where_arg = np.array(
        [
            [
                [
                    [[False, False], [False, False]],
                    [[True, True], [True, True]],
                    [[False, False], [False, False]],
                    [[True, True], [True, True]],
                    [[False, False], [False, False]],
                ],
                [
                    [[False, False], [False, False]],
                    [[True, True], [True, True]],
                    [[False, False], [False, False]],
                    [[True, True], [True, True]],
                    [[False, False], [False, False]],
                ],
            ]
        ]
    )

    where_mock.assert_called_once()
    where_call_args = where_mock.call_args.args
    np.testing.assert_array_equal(where_call_args[0].data, expected_where_arg)
    assert where_call_args[1] == 0.0
    assert where_call_args[2] is gaintable_mock["gain"]

    assert apply_ufunc_mock.call_count == 2

    gaintable_mock.assign.assert_called_once()
    assign_call_args = gaintable_mock.assign.call_args.args
    assign_dict = assign_call_args[0]
    assert assign_dict["gain"] == "NEW_GAIN"
    np.testing.assert_array_equal(assign_dict["weight"].data, expected_weights)

    gaintable_mock.chunk.assert_has_calls(
        [mock.call({"frequency": -1}), mock.call(gaintable_mock.chunks)]
    )

    assert fits["amp_fit"].shape == (1, nstations, nfreq, 2, 2)
    assert fits["phase_fit"].shape == (1, nstations, nfreq, 2, 2)
    assert fits["real_fit"].shape == (1, nstations, nfreq, 2, 2)
    assert fits["imag_fit"].shape == (1, nstations, nfreq, 2, 2)

    np.testing.assert_allclose(fits["amp_fit"][0, :, :, 0, 0], amp_fit_1.data)
    np.testing.assert_allclose(
        fits["phase_fit"][0, :, :, 0, 0], phase_fit_1.data
    )
    np.testing.assert_allclose(
        fits["real_fit"][0, :, :, 0, 0], real_fit_1.data
    )
    np.testing.assert_allclose(
        fits["imag_fit"][0, :, :, 0, 0], imag_fit_1.data
    )


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors.gain_flagging"
    ".GainFlagger"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors.gain_flagging"
    ".xr.apply_ufunc"
)
def test_should_perform_gain_flagging_without_apply(
    apply_ufunc_mock, gain_flagger_mock
):
    soltype = "real-imag"
    mode = "smooth"
    order = 1
    n_sigma = 0.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 2

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
    weights.copy.return_value = weights
    weights.data = np.ones((1, nstations, nfreq, 2, 2))

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
    gaintable_mock.weight = weights
    gaintable_mock.weights = weights

    dims_flag = ("antenna", "frequency")
    coords_flag = {"antenna": antenna_coords, "frequency": freq_coords}

    apply_ufunc_mock.side_effect = [
        (
            xr.DataArray(
                np.array([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]]),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
        (
            xr.DataArray(
                np.array([[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]]),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
        (
            xr.DataArray(
                np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]]),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
        (
            xr.DataArray(
                np.array([[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]]),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
    ]

    gain_flagger_mock.return_value = gain_flagger_mock

    gaintable, fits = flag_on_gains(
        gaintable_mock,
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains=False,
        skip_cross_pol=False,
        apply_flag=False,
    )

    assert apply_ufunc_mock.call_count == 4
    gaintable.assign.assert_called_once()
    assign_call_args = gaintable.assign.call_args.args
    assign_dict = assign_call_args[0]
    assert "weight" in assign_dict

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
    assert np.all(assign_dict["weight"].data == expected_weights)

    assert fits["amp_fit"].shape == gaintable_mock.gain.shape
    assert fits["phase_fit"].shape == gaintable_mock.gain.shape
    assert fits["real_fit"].shape == gaintable_mock.gain.shape
    assert fits["imag_fit"].shape == gaintable_mock.gain.shape

    gaintable.chunk.assert_has_calls(
        [mock.call({"frequency": -1}), mock.call(gaintable.chunks)]
    )
