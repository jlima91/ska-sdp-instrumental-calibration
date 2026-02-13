"""Utility functions for flux density calculations."""

from typing import Optional

import numpy as np


def calculate_flux_for_spectral_indices(
    flux: float,
    freq: np.ndarray,
    ref_freq: float,
    spec_idx: Optional[list],
    log_spec_idx: bool = True,
) -> np.ndarray:
    if spec_idx is None:
        spec_idx = []

    if log_spec_idx:
        return _logathmic_polynomial_method(flux, freq, ref_freq, spec_idx)

    return _linear_polynomial_method(flux, freq, ref_freq, spec_idx)


def _logathmic_polynomial_method(flux, freq, ref_freq, spec_idx):
    ratio = freq / ref_freq
    log_ratio = np.log10(ratio)

    # Reversing the spectral index list to match the order expected by
    # np.polyval
    exponent = np.polyval(spec_idx[::-1], log_ratio)
    return flux * (ratio**exponent)


def _linear_polynomial_method(flux, freq, ref_freq, spec_idx):
    ratio = freq / ref_freq

    # Reversing the spectral index list to match the order expected by
    # np.polyval and adding the flux as the constant term
    full_coeffs = spec_idx[::-1] + [flux]
    return np.polyval(full_coeffs, ratio - 1)
