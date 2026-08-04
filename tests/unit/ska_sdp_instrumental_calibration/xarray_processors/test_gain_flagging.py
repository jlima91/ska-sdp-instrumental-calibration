import numpy as np
import pytest
import xarray as xr
from mock import patch

from ska_sdp_instrumental_calibration.xarray_processors.gain_flagging import (
    GainFlagger,
    PhasorPolyFit,
    flag_on_gains,
)


def test_should_flag_gains_for_amplitude():
    soltype = "amplitude"
    order = 2
    n_sigma = 3.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3

    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    init_gains = gains.copy()
    gains[5] = 0 + 100j
    weights = np.ones(10)

    expected_weights = weights.copy()
    expected_weights[5] = 0
    expected_amp_fit = np.abs(init_gains)

    flagger_obj = GainFlagger(
        soltype=soltype,
        order=order,
        max_ncycles=max_ncycles,
        n_sigma=n_sigma,
        n_sigma_rolling=n_sigma_rolling,
        window_size=window_size,
        freq=frequencies,
    )

    flags, fits = flagger_obj.flag_dimension(gains, weights, None, None, None)

    updated_weights = weights.copy()
    updated_weights[flags] = 0

    np.testing.assert_allclose(
        fits["amp_fit"], expected_amp_fit, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(updated_weights, expected_weights)


def test_should_flag_gains_for_both_phase_and_amplitude():
    soltype = "amp-phase"
    order = 2
    n_sigma = 4
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3

    frequencies = np.arange(0, 1, 0.1)

    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    init_gains = gains.copy()
    gains[5] = -200  # Outlier in both amp and phase
    gains[4] = -500j
    weights = np.ones(10)

    expected_weights = weights.copy()
    expected_weights[4] = 0
    expected_weights[5] = 0

    flagger_obj = GainFlagger(
        soltype=soltype,
        order=order,
        max_ncycles=max_ncycles,
        n_sigma=n_sigma,
        n_sigma_rolling=n_sigma_rolling,
        window_size=window_size,
        freq=frequencies,
    )

    flags, fits = flagger_obj.flag_dimension(gains, weights, None, None, None)

    updated_weights = weights.copy()
    updated_weights[flags] = 0

    expected_amp_fit = np.abs(init_gains)

    expected_phase_fit = np.angle(init_gains)

    np.testing.assert_allclose(updated_weights, expected_weights)
    np.testing.assert_allclose(
        fits["amp_fit"], expected_amp_fit, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        fits["phase_fit"], expected_phase_fit, rtol=1e-5, atol=1e-6
    )


def test_should_flag_gains_for_real_imag():

    soltype = "real-imag"
    order = 2
    n_sigma = 4.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    gains[5] = 100 + 100j

    weights = np.ones(10)

    expected_weights = weights.copy()
    expected_weights[5] = 0

    flagger_obj = GainFlagger(
        soltype,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
    )

    flags, fits = flagger_obj.flag_dimension(
        gains,
        weights,
        antenna_name="a1",
        receptor1_name="X",
        receptor2_name="Y",
    )

    updated_weights = weights.copy()
    updated_weights[flags] = 0

    np.testing.assert_allclose(updated_weights, expected_weights)

    assert "real_fit" in fits
    assert "imag_fit" in fits
    assert "amp_fit" not in fits
    assert "phase_fit" not in fits

    real_fit = fits["real_fit"]
    imag_fit = fits["imag_fit"]

    assert real_fit.shape == gains.shape
    assert imag_fit.shape == gains.shape

    assert np.all(np.isfinite(real_fit[1:-1]))
    assert np.all(np.isfinite(imag_fit[1:-1]))

    np.testing.assert_allclose(
        real_fit[1:-1],
        np.nanmedian(
            np.vstack([gains.real[:-2], gains.real[1:-1], gains.real[2:]]),
            axis=0,
        ),
        rtol=1e-1,
        atol=1e-1,
    )


def test_should_throw_exception_if_nsigma_is_less_than_or_equal_to_zero():
    soltype = "real-imag"
    order = 1
    n_sigma = 0.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 2
    frequencies = np.arange(0, 1, 0.1)
    with pytest.raises(ValueError, match="n_sigma must be greater than zero"):
        GainFlagger(
            soltype,
            order,
            max_ncycles,
            n_sigma,
            n_sigma_rolling,
            window_size,
            frequencies,
        )


def test_should_throw_exception_if_window_size_is_even():
    soltype = "real-imag"
    order = 1
    n_sigma = 3.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 2
    frequencies = np.arange(0, 1, 0.1)
    with pytest.raises(ValueError, match="window_size must be odd"):
        GainFlagger(
            soltype,
            order,
            max_ncycles,
            n_sigma,
            n_sigma_rolling,
            window_size,
            frequencies,
        )


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "gain_flagging.xr.apply_ufunc"
)
def test_should_perform_gain_flagging(apply_ufunc_mock):

    soltype = "amplitude"

    order = 1
    n_sigma = 3.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 3

    nstations = 2
    nfreq = 5

    gain_data = np.ones((1, nstations, nfreq, 2, 2)) + 1j
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

    gaintable = xr.Dataset(
        {
            "gain": xr.DataArray(gain_data, coords=coords, dims=dims),
            "weight": xr.DataArray(
                np.ones((1, nstations, nfreq, 2, 2)),
                coords=coords,
                dims=dims,
            ),
            "configuration": xr.DataArray(
                np.arange(nstations),
                dims=("id",),
                coords={"names": ("id", antenna_coords)},
            ),
        }
    ).chunk({"frequency": 2})

    dims2 = ("time", "antenna", "frequency")
    coords2 = {
        "time": [0],
        "antenna": antenna_coords,
        "frequency": freq_coords,
    }

    weight_flag_1 = xr.DataArray(
        [[[False, False, False, True, False], [False, False, False, True, False]]],
        dims=dims2,
        coords=coords2,
    )

    amp_fit_1 = xr.DataArray(
        [[[0.1, 0.1, 0.1, 0.0, 0.1], [0.1, 0.1, 0.1, 0.0, 0.1]]],
        dims=dims2,
        coords=coords2,
    )

    weight_flag_2 = xr.DataArray(
        [[[False, True, False, False, False], [False, True, False, False, False]]],
        dims=dims2,
        coords=coords2,
    )

    amp_fit_2 = xr.DataArray(
        [[[0.3, 0.0, 0.3, 0.3, 0.3], [0.3, 0.0, 0.3, 0.3, 0.3]]],
        dims=dims2,
        coords=coords2,
    )

    apply_ufunc_mock.side_effect = [
        (weight_flag_1, amp_fit_1),
        (weight_flag_2, amp_fit_2),
    ]

    result_gaintable, fits = flag_on_gains(
        gaintable,
        soltype,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains=False,
        skip_cross_pol=True,
        apply_flag=True,
    )

    assert apply_ufunc_mock.call_count == 2

    assert "amp_fit" in fits
    assert "phase_fit" not in fits
    assert "real_fit" not in fits
    assert "imag_fit" not in fits

    assert fits["amp_fit"].shape == (1, nstations, nfreq, 2, 2)

    np.testing.assert_allclose(
        fits["amp_fit"][0, :, :, 0, 0].data,
        amp_fit_1.data[0],
    )
    np.testing.assert_allclose(
        fits["amp_fit"][0, :, :, 1, 1].data,
        amp_fit_2.data[0],
    )
    np.testing.assert_allclose(fits["amp_fit"][..., 0, 1].data, 0.0)
    np.testing.assert_allclose(fits["amp_fit"][..., 1, 0].data, 0.0)

    expected_weight = np.ones((1, nstations, nfreq, 2, 2), dtype=np.float64)
    expected_weight[:, :, 3, 0, 0] = 0.0
    expected_weight[:, :, 1, 1, 1] = 0.0
    np.testing.assert_allclose(result_gaintable.weight.data, expected_weight)

    expected_gain = np.ones((1, nstations, nfreq, 2, 2), dtype=np.complex128) + 1j
    expected_gain[:, :, 3, 0, 0] = 0.0j
    expected_gain[:, :, 1, 1, 1] = 0.0j
    np.testing.assert_allclose(result_gaintable.gain.data, expected_gain)


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors.gain_flagging"
    ".xr.apply_ufunc"
)
def test_should_perform_gain_flagging_without_apply(
    apply_ufunc_mock,
):
    soltype = "real-imag"
    order = 1
    n_sigma = 3.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 3

    nstations = 2
    nfreq = 5

    gain_data = np.ones((1, nstations, nfreq, 2, 2)) + 1j
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

    gaintable = xr.Dataset(
        {
            "gain": xr.DataArray(gain_data, coords=coords, dims=dims),
            "weight": xr.DataArray(
                np.ones((1, nstations, nfreq, 2, 2)),
                coords=coords,
                dims=dims,
            ),
            "configuration": xr.DataArray(
                np.arange(nstations),
                dims=("id",),
                coords={"names": ("id", antenna_coords)},
            ),
        }
    ).chunk({"frequency": 2})
    original_chunks = gaintable.chunksizes

    dims_flag = ("time", "antenna", "frequency")
    coords_flag = {
        "time": [0],
        "antenna": antenna_coords,
        "frequency": freq_coords,
    }

    apply_ufunc_mock.side_effect = [
        (
            xr.DataArray(
                np.array(
                    [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]],
                    dtype=bool,
                ).reshape(1, nstations, nfreq),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
        (
            xr.DataArray(
                np.array(
                    [[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]],
                    dtype=bool,
                ).reshape(1, nstations, nfreq),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
        (
            xr.DataArray(
                np.array(
                    [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]],
                    dtype=bool,
                ).reshape(1, nstations, nfreq),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
        (
            xr.DataArray(
                np.array(
                    [[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]],
                    dtype=bool,
                ).reshape(1, nstations, nfreq),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
            xr.DataArray(
                np.zeros((1, nstations, nfreq)),
                dims=dims_flag,
                coords=coords_flag,
            ),
        ),
    ]

    gaintable, fits = flag_on_gains(
        gaintable,
        soltype,
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

    expected_weights = np.array(
        [
            [
                [
                    [[1, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [1, 0]],
                    [[0, 0], [0, 1]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[1, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [1, 0]],
                    [[0, 0], [0, 1]],
                    [[0, 0], [0, 0]],
                ],
            ]
        ]
    )

    np.testing.assert_array_equal(gaintable.weight.data, expected_weights)
    np.testing.assert_allclose(gaintable.gain.data, gain_data)

    assert "real_fit" in fits
    assert "imag_fit" in fits
    assert "amp_fit" not in fits
    assert "phase_fit" not in fits

    assert fits["real_fit"].shape == gaintable.gain.shape
    assert fits["imag_fit"].shape == gaintable.gain.shape
    assert gaintable.chunksizes == original_chunks


def test_phasor_polyfit_computes_freq_guess_when_none():

    freq = np.linspace(0, 1, 50)
    true_freq = 5.0
    gains = np.exp(1j * 2 * np.pi * true_freq * freq)
    flags = np.zeros_like(freq, dtype=bool)
    fitter = PhasorPolyFit(order=1, freq=freq)
    model, estimated_freq = fitter.fit(
        gains,
        flags,
        freq_guess=None,
    )

    assert estimated_freq is not None
    assert np.isclose(estimated_freq, true_freq, atol=1.0)
    assert np.all(np.isfinite(model))


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "gain_flagging.logger"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "gain_flagging.curve_fit"
)
def test_phasor_polyfit_runtime_error_logged(curve_fit_mock, logger_mock):

    curve_fit_mock.side_effect = RuntimeError("fit failed")

    freq = np.linspace(0, 1, 20)
    gains = np.ones_like(freq, dtype=complex)
    flags = np.zeros_like(freq, dtype=bool)

    fitter = PhasorPolyFit(order=1, freq=freq)

    model, freq_guess = fitter.fit(gains, flags, freq_guess=None)

    curve_fit_mock.assert_called_once()

    logger_mock.warning.assert_called_once_with(
        "Phasor fit failed, returning NaNs"
    )
    assert np.all(np.isnan(model))


def test_gain_flagger_smooth_branch_executes():

    soltype = "real-imag"
    order = 3
    n_sigma = 5.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3
    freq = np.linspace(0, 1, 20)
    gains = np.linspace(1, 2, 20) + 1j * np.linspace(2, 1, 20)
    weights = np.ones_like(gains)

    flagger = GainFlagger(
        soltype,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        freq,
    )

    flagged_weights, fits = flagger.flag_dimension(
        gains,
        weights,
        antenna_name="a1",
        receptor1_name="X",
        receptor2_name="Y",
    )

    assert "real_fit" in fits
    assert "imag_fit" in fits

    assert fits["real_fit"].shape == gains.shape
    assert fits["imag_fit"].shape == gains.shape

    assert np.all(np.isfinite(fits["real_fit"][1:-1]))
    assert np.all(np.isfinite(fits["imag_fit"][1:-1]))
