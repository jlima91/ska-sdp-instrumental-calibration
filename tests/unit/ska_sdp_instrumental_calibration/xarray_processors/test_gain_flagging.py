import mock
import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.xarray_processors.gain_flagging import (
    GainFlagger,
    PhasorPolyFit,
    flag_on_gains,
)


def test_should_flag_gains_for_amplitude():
    soltype = "amplitude"
    mode = "poly"
    order = 3
    n_sigma = 2.0
    max_ncycles = 1
    n_sigma_rolling = 0.0
    window_size = 3

    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    gains[5] = 0 + 100j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype=soltype,
        mode=mode,
        order=order,
        max_ncycles=max_ncycles,
        n_sigma=n_sigma,
        n_sigma_rolling=n_sigma_rolling,
        window_size=window_size,
        freq=frequencies,
    )

    updated_weights, fits = flagger_obj.flag_dimension(gains, weights)

    expected_weights = np.array(
        [
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
    )
    expected_amp_fit = np.array(
        [
            2.814829,
            2.06567,
            8.526082,
            15.493124,
            21.533434,
            25.266551,
            25.317458,
            20.319706,
            8.989625,
            10.724825,
        ]
    )

    np.testing.assert_allclose(updated_weights, expected_weights)
    np.testing.assert_allclose(
        fits["amp_fit"], expected_amp_fit, rtol=1e-5, atol=1e-6
    )


def test_should_flag_gains_for_both_phase_and_amplitude():
    soltype = "amp-phase"
    mode = "poly"
    order = 3
    n_sigma = 3
    max_ncycles = 1
    n_sigma_rolling = 3.0
    window_size = 3

    frequencies = np.arange(0, 1, 0.1)

    gains = np.zeros(10, dtype=complex)
    gains[5] = -200  # Outlier in both amp and phase
    gains[4] = -500j

    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype=soltype,
        mode=mode,
        order=order,
        max_ncycles=max_ncycles,
        n_sigma=n_sigma,
        n_sigma_rolling=n_sigma_rolling,
        window_size=window_size,
        freq=frequencies,
    )

    updated_weights, fits = flagger_obj.flag_dimension(gains, weights)

    expected_weights = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )

    expected_amp_fit = np.array(
        [
            66.713259,
            39.813524,
            110.442876,
            149.370605,
            160.792513,
            148.904404,
            117.90208,
            71.981342,
            15.337996,
            47.832161,
        ]
    )

    expected_phase_fit = np.array(
        [
            1.576405e00,
            -3.137745e00,
            1.573614e00,
            1.609824e-03,
            -1.570416e00,
            3.140736e00,
            1.568704e00,
            -3.314318e-03,
            -1.575081e00,
            -6.030979e-03,
        ]
    )

    np.testing.assert_allclose(updated_weights, expected_weights)
    np.testing.assert_allclose(
        fits["amp_fit"], expected_amp_fit, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        fits["phase_fit"], expected_phase_fit, rtol=1e-5, atol=1e-6
    )


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
    gains[5] = 100 + 100j

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
    )

    flagged_weights, fits = flagger_obj.flag_dimension(
        gains,
        weights,
        antenna="a1",
        receptor1="X",
        receptor2="Y",
    )

    assert flagged_weights[5] == 0.0

    assert np.sum(flagged_weights == 0) >= 1

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
        rtol=1e-2,
        atol=1e-2,
    )


def test_should_throw_exception_if_window_size_is_even():

    soltype = "real-imag"
    mode = "smooth"
    order = 1
    n_sigma = 0.0
    max_ncycles = 1
    n_sigma_rolling = 15.0
    window_size = 2
    frequencies = np.arange(0, 1, 0.1)
    with pytest.raises(ValueError, match="window_size must be odd"):
        GainFlagger(
            soltype,
            mode,
            order,
            max_ncycles,
            n_sigma,
            n_sigma_rolling,
            window_size,
            frequencies,
        )


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
    mode = "poly"

    order = 1
    n_sigma = 0.0
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

    gaintable_mock = MagicMock(name="gaintable")
    gaintable_mock.chunk.return_value = gaintable_mock
    gaintable_mock.assign.return_value = gaintable_mock
    gaintable_mock.chunks = {"frequency": 2}

    gaintable_mock.receptor1 = xr.DataArray(["X", "Y"], dims="id")
    gaintable_mock.receptor2 = xr.DataArray(["X", "Y"], dims="id")

    gaintable_mock.gain = xr.DataArray(gain_data, coords=coords, dims=dims)

    gaintable_mock.weight = xr.DataArray(
        np.ones((1, nstations, nfreq, 2, 2)),
        coords=coords,
        dims=dims,
    )

    dims2 = ("antenna", "frequency")
    coords2 = {"antenna": antenna_coords, "frequency": freq_coords}

    weight_flag_1 = xr.DataArray(
        [[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]],
        dims=dims2,
        coords=coords2,
    )

    amp_fit_1 = xr.DataArray(
        [[0.1, 0.1, 0.1, 0.0, 0.1], [0.1, 0.1, 0.1, 0.0, 0.1]],
        dims=dims2,
        coords=coords2,
    )

    weight_flag_2 = xr.DataArray(
        [[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]],
        dims=dims2,
        coords=coords2,
    )

    amp_fit_2 = xr.DataArray(
        [[0.3, 0.0, 0.3, 0.3, 0.3], [0.3, 0.0, 0.3, 0.3, 0.3]],
        dims=dims2,
        coords=coords2,
    )

    apply_ufunc_mock.side_effect = [
        (weight_flag_1, amp_fit_1),
        (weight_flag_2, amp_fit_2),
    ]

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

    assert apply_ufunc_mock.call_count == 2
    where_mock.assert_called_once()
    gaintable_mock.assign.assert_called()

    assert "amp_fit" in fits
    assert "phase_fit" not in fits
    assert "real_fit" not in fits
    assert "imag_fit" not in fits

    assert fits["amp_fit"].shape == (1, nstations, nfreq, 2, 2)

    np.testing.assert_allclose(
        fits["amp_fit"][0, :, :, 0, 0],
        amp_fit_1.data,
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

    gaintable_mock = MagicMock(name="gaintable")
    gaintable_mock.chunk.return_value = gaintable_mock
    gaintable_mock.assign.return_value = gaintable_mock
    gaintable_mock.chunks = {"frequency": 2}

    gaintable_mock.receptor1 = xr.DataArray(["X", "Y"], dims="id")
    gaintable_mock.receptor2 = xr.DataArray(["X", "Y"], dims="id")

    gaintable_mock.gain = xr.DataArray(gain_data, coords=coords, dims=dims)

    gaintable_mock.weight = xr.DataArray(
        np.ones((1, nstations, nfreq, 2, 2)),
        coords=coords,
        dims=dims,
    )

    dims_flag = ("antenna", "frequency")
    coords_flag = {"antenna": antenna_coords, "frequency": freq_coords}

    apply_ufunc_mock.side_effect = [
        (
            xr.DataArray(
                [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]],
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
                [[1, 0, 1, 1, 1], [1, 0, 1, 1, 1]],
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
                [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]],
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
                [[1, 1, 1, 0, 1], [1, 1, 1, 0, 1]],
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

    assert gaintable_mock.assign.call_count == 1

    assign_kwargs = gaintable_mock.assign.call_args.kwargs
    assert "weight" in assign_kwargs

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

    np.testing.assert_array_equal(
        assign_kwargs["weight"].data,
        expected_weights,
    )

    assert "real_fit" in fits
    assert "imag_fit" in fits
    assert "amp_fit" not in fits
    assert "phase_fit" not in fits

    assert fits["real_fit"].shape == gaintable_mock.gain.shape
    assert fits["imag_fit"].shape == gaintable_mock.gain.shape

    gaintable_mock.chunk.assert_has_calls(
        [
            mock.call({"frequency": -1}),
            mock.call(gaintable.chunks),
        ]
    )


def test_phasor_polyfit_computes_freq_guess_when_none():

    freq = np.linspace(0, 1, 50)
    true_freq = 5.0
    gains = np.exp(1j * 2 * np.pi * true_freq * freq)
    weights = np.ones_like(gains)

    fitter = PhasorPolyFit(order=1, freq=freq)
    model, estimated_freq = fitter.fit(
        gains,
        weights,
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
    weights = np.ones_like(freq)

    fitter = PhasorPolyFit(order=1, freq=freq)

    model, freq_guess = fitter.fit(gains, weights, freq_guess=None)

    curve_fit_mock.assert_called_once()

    logger_mock.warning.assert_called_once_with(
        "Phasor fit failed, returning NaNs"
    )
    assert np.all(np.isnan(model))


def test_gain_flagger_smooth_branch_executes():

    soltype = "real-imag"
    mode = "smooth"
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
        mode,
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
        antenna="a1",
        receptor1="X",
        receptor2="Y",
    )

    assert "real_fit" in fits
    assert "imag_fit" in fits

    assert fits["real_fit"].shape == gains.shape
    assert fits["imag_fit"].shape == gains.shape

    assert np.all(np.isfinite(fits["real_fit"][1:-1]))
    assert np.all(np.isfinite(fits["imag_fit"][1:-1]))
