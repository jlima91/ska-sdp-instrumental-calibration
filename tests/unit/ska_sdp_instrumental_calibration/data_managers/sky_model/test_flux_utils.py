import numpy as np

from ska_sdp_instrumental_calibration.data_managers.sky_model import flux_utils

calculate_flux_for_spectral_indices = (
    flux_utils.calculate_flux_for_spectral_indices
)


def test_log_model_constant_spectral_index_vectorised():
    flux = 2.5
    ref_freq = 1.0e8
    freq = np.array([0.5e8, 1.0e8, 2.0e8])
    spec_idx = [-0.7]

    out = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx, True
    )
    expected = flux * (freq / ref_freq) ** (-0.7)
    assert np.allclose(out, expected)


def test_use_logarthmic_polynomial_method_when_log_spec_id_is_true():
    flux = 1.0
    ref_freq = 1.0e8
    freq = np.array([0.5e8, 1.0e8, 2.0e8])

    # spec_idx encodes exponent(log10(ratio)) = a0 + a1*log10(ratio)
    a0, a1, a2, a3 = -0.2, 0.05, 0.6, 0.78
    spec_idx = [a0, a1, a2, a3]

    ratio = freq / ref_freq
    log_ratio = np.log10(ratio)
    exponent = a0 + a1 * log_ratio + a2 * log_ratio**2 + a3 * log_ratio**3
    expected = flux * (ratio**exponent)

    out = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx, log_spec_idx=True
    )
    assert np.allclose(out, expected)


def test_use_linear_polynomial_method_when_log_spec_id_is_false():
    flux = 10.0
    ref_freq = 100.0
    freq = np.array([50.0, 100.0, 200.0])
    spec_idx = [-0.7]

    ratio = freq / ref_freq
    expected = flux + (-0.7) * (ratio - 1.0)

    out = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx, False
    )
    assert np.allclose(out, expected)


def test_if_spec_idx_is_none_then_return_reference_flux_log_and_linear():
    flux = 3.3
    ref_freq = 1.0e8
    freq = np.array([0.8e8, 1.0e8, 1.2e8])

    out_log = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx=None, log_spec_idx=True
    )
    out_lin = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx=None, log_spec_idx=False
    )
    expected = np.full(shape=len(freq), fill_value=flux)

    assert np.allclose(out_log, expected)
    assert np.allclose(out_lin, expected)


def test_if_spec_idx_is_empty_list_returns_reference_flux_log_and_linear():
    flux = 3.3
    ref_freq = 1.0e8
    freq = np.array([0.8e8, 1.0e8, 1.2e8])

    out_log = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx=[], log_spec_idx=True
    )
    out_lin = calculate_flux_for_spectral_indices(
        flux, freq, ref_freq, spec_idx=[], log_spec_idx=False
    )

    expected = np.full(shape=len(freq), fill_value=flux)
    assert np.allclose(out_log, expected)
    assert np.allclose(out_lin, expected)
