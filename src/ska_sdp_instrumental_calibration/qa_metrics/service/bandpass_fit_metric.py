"""
Bandpass fit metrics of a field: the fit quality of the phase and of the
gain amplitude of each station, in X and Y polarisation.

A fit quality is the ratio of channels whose distance to the fit is
within ``n_sigma`` robust standard deviations (σ = IQR / 1.349) to all
fitted channels, pooled over the times. Flagged channels (weight 0) are
left out of both counts: they are reported by the flagging statistics.
"""

import numpy as np

from ...data_managers.schema.calibration_set import GainCalibrationSetXds
from ..qa_section import QASection
from ..run_metadata import FieldMetadata

SKA_LOW_COARSE_CHANNEL_WIDTH = 781250.0  # Hz

# Ratio of the interquartile range to the standard deviation of a normal
# distribution
IQR_TO_SIGMA = 1.349


def count_within_n_sigma(residuals: np.ndarray, n_sigma: float) -> int:
    """
    Count the residuals within n_sigma robust standard deviations of 0.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of a fit, one per channel.
    n_sigma : float
        Distance to the fit, in robust standard deviations, within which
        a channel is good.

    Returns
    -------
    int
        Number of good channels.
    """
    # Robust standard deviation from the IQR: unlike the plain standard
    # deviation, it is barely inflated by the outliers it must detect
    q25, q75 = np.percentile(residuals, [25, 75])
    sigma = (q75 - q25) / IQR_TO_SIGMA

    # "<=" so that a perfect fit (sigma == 0) counts all channels as good
    return int((np.abs(residuals) <= n_sigma * sigma).sum())


def count_good_phase_channels(
    nu: np.ndarray, phase: np.ndarray, n_sigma: float
) -> int:
    """
    Count the channels close to a joint linear delay + 1/ν TEC fit of the
    phase across the band.

    The phase is modelled as ``φ(ν) = a + b·ν + c/ν``: a constant offset,
    a linear delay term and a 1/ν dispersive (TEC) term. The model is
    linear in its parameters, so it is fitted by least squares.

    Parameters
    ----------
    nu : numpy.ndarray
        Frequencies of the usable channels, in Hz, in ascending order.
    phase : numpy.ndarray
        Unwrapped phase of the usable channels.
    n_sigma : float
        Distance to the fit, in robust standard deviations, within which
        a channel is good.

    Returns
    -------
    int
        Number of good channels.
    """
    # Frequencies scaled to the band centre keep the design matrix well
    # conditioned: ν and 1/ν in Hz differ by ~16 orders of magnitude
    nu_0 = np.median(nu)
    design = np.column_stack([np.ones_like(nu), nu / nu_0, nu_0 / nu])
    params, *_ = np.linalg.lstsq(design, phase, rcond=None)

    # Residuals wrapped into (-π, π], so a channel that unwrapped onto the
    # wrong turn is not counted as 2π away from the fit
    residuals = np.angle(np.exp(1j * (phase - design @ params)))
    return count_within_n_sigma(residuals, n_sigma)


def coarse_channel_amplitude_residuals(
    nu: np.ndarray,
    amplitude: np.ndarray,
    coarse: np.ndarray,
    min_coarse_chan: int = 3,
) -> np.ndarray:
    """
    Residuals of 1st order polynomial fits of the gain amplitude, one fit
    per SKA-Low coarse channel.

    Parameters
    ----------
    nu : numpy.ndarray
        Frequencies of the usable channels, in Hz.
    amplitude : numpy.ndarray
        Gain amplitude of the usable channels.
    coarse : numpy.ndarray
        Coarse channel of each usable channel.
    min_coarse_chan : int
        Minimum number of usable channels of a coarse channel to fit it:
        a line through 2 channels fits them exactly.

    Returns
    -------
    numpy.ndarray
        Residuals of the channels of the fitted coarse channels. Empty if
        no coarse channel has enough channels.
    """
    residuals = []
    for channel in np.unique(coarse):
        in_channel = coarse == channel
        if in_channel.sum() < min_coarse_chan:
            continue

        nu_c = nu[in_channel]
        # Centred on the coarse channel, so the fit is well conditioned
        design = np.column_stack([np.ones_like(nu_c), nu_c - nu_c.mean()])
        params, *_ = np.linalg.lstsq(design, amplitude[in_channel], rcond=None)
        residuals.append(amplitude[in_channel] - design @ params)

    return np.concatenate(residuals) if residuals else np.empty(0)


def gain_fit_quality_for_pol(
    station, pol: str, min_fit_channels: int = 4, n_sigma: float = 5.0
) -> float | None:
    """
    Gain amplitude fit quality of a station in one polarisation.

    The residuals of all coarse channels of a time are pooled for one
    robust standard deviation, as a coarse channel has too few channels
    for a robust estimate of its own. Channels of coarse channels too
    few to fit are left out of both counts.

    Parameters
    ----------
    station : GainCalibrationSetXds
        Gaintable of one station, in ascending frequency.
    pol : str
        Polarisation, e.g. ``XX``.
    min_fit_channels : int
        Minimum number of usable channels of a time to fit it.
    n_sigma : float
        Distance to the fit, in robust standard deviations, within which
        a channel is good.

    Returns
    -------
    float or None
        Fit quality between 0 and 1, None if no time could be fitted.
    """
    frequency = station.frequency.values
    # Shape (time, frequency)
    gains_per_time = station.CALPARAM_GAIN.sel(polarization=pol).values
    weights_per_time = station.CALPARAM_WEIGHT.sel(polarization=pol).values

    total_good = 0
    total_fitted = 0
    for gains, weights in zip(gains_per_time, weights_per_time):
        usable = (weights != 0) & np.isfinite(gains) & (gains != 0)
        if usable.sum() < min_fit_channels:
            continue

        usable_frequency = frequency[usable]
        coarse_channel = np.round(
            usable_frequency / SKA_LOW_COARSE_CHANNEL_WIDTH
        ).astype(int)
        residuals = coarse_channel_amplitude_residuals(
            usable_frequency, np.abs(gains[usable]), coarse_channel
        )
        # Too few channels per coarse channel to fit any of them, e.g. a
        # gaintable at coarse channel resolution
        if residuals.size < min_fit_channels:
            continue

        total_good += count_within_n_sigma(residuals, n_sigma)
        # Only the fitted channels: the others cannot be judged
        total_fitted += residuals.size

    return total_good / total_fitted if total_fitted else None


def phase_fit_quality_for_pol(
    station, pol: str, min_fit_channels: int = 4, n_sigma: float = 5.0
) -> float | None:
    """
    Phase fit quality of a station in one polarisation.

    Parameters
    ----------
    station : GainCalibrationSetXds
        Gaintable of one station, in ascending frequency.
    pol : str
        Polarisation, e.g. ``XX``.
    min_fit_channels : int
        Minimum number of usable channels of a time to fit it; more than
        the 3 parameters of the delay + TEC model.
    n_sigma : float
        Distance to the fit, in robust standard deviations, within which
        a channel is good.

    Returns
    -------
    float or None
        Fit quality between 0 and 1, None if no time could be fitted.
    """
    frequency = station.frequency.values
    # Shape (time, frequency)
    gains_per_time = station.CALPARAM_GAIN.sel(polarization=pol).values
    weights_per_time = station.CALPARAM_WEIGHT.sel(polarization=pol).values

    total_good = 0
    total_fitted = 0
    for gains, weights in zip(gains_per_time, weights_per_time):
        usable = (weights != 0) & np.isfinite(gains) & (gains != 0)
        n_usable = int(usable.sum())
        if n_usable < min_fit_channels:
            continue

        # Unwrapping over the usable channels only, so a flagged gap does
        # not break the phase continuity
        phase = np.unwrap(np.angle(gains[usable]))
        total_good += count_good_phase_channels(
            frequency[usable], phase, n_sigma
        )
        total_fitted += n_usable

    return total_good / total_fitted if total_fitted else None


def bandpass_fit_quality(
    gaintable: GainCalibrationSetXds,
) -> dict[str, dict[str, float | None]]:
    """
    Get the phase and gain amplitude fit quality of each station in X and
    Y polarisation.

    The phase of each time is fitted with a joint linear delay + 1/ν TEC
    model across the band; the gain amplitude with a 1st order polynomial
    per SKA-Low coarse channel (781.25 kHz).

    Parameters
    ----------
    gaintable : GainCalibrationSetXds
        Bandpass gaintable of the field, with ``CALPARAM_GAIN`` and
        ``CALPARAM_WEIGHT`` of dimensions (time, antenna_name, frequency,
        polarization) and ``XX`` and ``YY`` polarisations.

    Returns
    -------
    dict[str, dict[str, float | None]]
        ``x_phase_fit_quality``, ``y_phase_fit_quality``,
        ``x_gains_fit_quality`` and ``y_gains_fit_quality``, between 0 and
        1, per MS station name of the gaintable. None where a station has
        too few channels to fit.
    """
    # Ascending frequency, as unwrapping needs neighbouring channels to be
    # adjacent; sorted once for all stations
    sorted_gaintable = gaintable.sortby("frequency")

    # Polarisation of the gaintable: prefix of its column keys
    column_prefixes = {"XX": "x", "YY": "y"}

    fit_quality_per_station = {}
    for station_name in sorted_gaintable.antenna_name.values:
        station_gaintable = sorted_gaintable.sel(antenna_name=station_name)

        fit_quality = {}
        for pol, prefix in column_prefixes.items():
            fit_quality[f"{prefix}_phase_fit_quality"] = (
                phase_fit_quality_for_pol(station_gaintable, pol)
            )
            fit_quality[f"{prefix}_gains_fit_quality"] = (
                gain_fit_quality_for_pol(station_gaintable, pol)
            )

        fit_quality_per_station[str(station_name)] = fit_quality

    return fit_quality_per_station


def get_bandpass_gaintable_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the bandpass gaintables of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one bandpass gaintable recorded
        per MS prefix.

    Returns
    -------
    list[str]
        Bandpass gaintable of each measurement set of the field. Empty if
        none is recorded: the reader raises for it.
    """
    gaintables = (
        field_metadata.get_data_for(QASection.BANDPASS.GAINTABLE) or {}
    )
    return list(gaintables.values())
