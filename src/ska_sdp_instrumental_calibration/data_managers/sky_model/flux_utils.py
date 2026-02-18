import numpy as np


def calculate_flux_for_spectral_indices(
    flux: float,
    freq: np.ndarray,
    ref_freq: float,
    spec_idx: list,
    log_spec_idx: bool,
) -> np.ndarray:
    """
    Calculate flux at given frequencies using spectral indices.

    This function computes flux using either a logarithmic or linear
    polynomial spectral model. The model type is controlled by the
    `log_spec_idx` parameter.

    Parameters
    ----------
    flux
        Reference flux (Jy) at the reference frequency.
    freq
        Frequencies at which to calculate flux (Hz).
    ref_freq
        Reference frequency (Hz).
    spec_idx
        Spectral index polynomial coefficients.
    log_spec_idx
        If True (default), use logarithmic polynomial method. If False, use
        linear polynomial method.

    Returns
    -------
        Flux at the specified frequencies. Same shape as `freq`.
    """
    if spec_idx is None:
        spec_idx = []

    if log_spec_idx:
        return _logarthmic_polynomial_method(flux, freq, ref_freq, spec_idx)

    return _linear_polynomial_method(flux, freq, ref_freq, spec_idx)


def _logarthmic_polynomial_method(flux, freq, ref_freq, spec_idx):
    ratio = freq / ref_freq
    log_ratio = np.log10(ratio)

    # Reversing the spectral index list to match the order expected
    # by np.polyval
    exponent = np.polyval(spec_idx[::-1], log_ratio)
    return flux * (ratio**exponent)


def _linear_polynomial_method(flux, freq, ref_freq, spec_idx):
    ratio = freq / ref_freq

    # Reversing the spectral index list to match the order expected by
    # np.polyval and adding the flux as the constant term
    full_coeffs = spec_idx[::-1] + [flux]
    return np.polyval(full_coeffs, ratio - 1)
