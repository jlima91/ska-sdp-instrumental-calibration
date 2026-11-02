#!/usr/bin/env python3
"""
Gain table generation script for radio interferometry simulations.

Authored by:
- Maciej Serylak
- Team Dhruva

The following effects can be simulated:
- Bandpass amplitude shape across frequency, fitted to
  measured data and perturbed across stations.
- Bandpass phase slopes across frequency (Bezier curves)
- Slow time-varying gain and phase fluctuations, simulated as sinusoidal
  functions (e.g. amplitude variations with ~2.4 h period, phase variations
  with ~0.9 h period).
- Random per-station offsets in amplitude and phase.
- RFI (radio-frequency interference), injected as a large amplitude
  perturbation within a specified frequency band.
- Gain outliers for random stations and channels.

The script needs a YAML config file as input, which contains the necessary
parameters.

The generated gain solutions are written to an OSKAR gain model HDF5 file
(gain_model.h5, see lib/gaintable/h5parm_from_oskar_gains.py to convert it
to H5parm) with datasets:
- "freq (Hz)"   : frequency axis in Hz
- "gain_xpol"   : complex gain solutions for X polarization
- "gain_ypol"   : complex gain solutions for Y polarization

Usage:
python generate_gaintable.py sim.yaml
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from lib.sim_config import SimConfig

# Fixed seed, in order to achieve repeatability of results.
RANDOM_SEED = 100

# Arbitrary control points of the Bezier curves of the bandpass phase,
# per polarisation.
BANDPASS_PHASE_CONTROL_POINTS = {
    "X": ([0, 6], [0.2, -1], [0.05, -3], [1, -5]),
    "Y": ([0, 6.1], [0.1, -1.2], [0.05, -3.2], [1, -5.3]),
}

N_POLS = 2


@dataclass
class GainAxes:
    """Frequency (MHz), time (s) and station axes of the gain table."""

    start_freq_mhz: float
    end_freq_mhz: float
    freqs_mhz: np.ndarray
    times_s: np.ndarray
    sampling_time_s: float
    n_stations: int

    @property
    def n_times(self) -> int:
        return len(self.times_s)

    @classmethod
    def from_config(cls, cfg: SimConfig) -> "GainAxes":
        start_freq_mhz = cfg.simulation_start_frequency_hz * 1e-6
        end_freq_mhz = cfg.simulation_end_frequency_hz * 1e-6
        channel_width_mhz = cfg.correlated_channel_bandwidth_hz * 1e-6
        sampling_time_s = cfg.sampling_time_sec
        n_times = int(cfg.observing_time_mins * 60 // sampling_time_s)
        return cls(
            start_freq_mhz=start_freq_mhz,
            end_freq_mhz=end_freq_mhz,
            # TODO: Verify what is the correct way to call arange
            freqs_mhz=np.arange(start_freq_mhz, end_freq_mhz, channel_width_mhz),
            # TODO: Verify what is the correct way to call arange
            times_s=np.arange(0, n_times * sampling_time_s, sampling_time_s),
            sampling_time_s=sampling_time_s,
            n_stations=cfg.n_stations,
        )


def load_config(yaml_file) -> SimConfig:
    return SimConfig.load(yaml_file)


def seed_random_generators(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def create_gaussian_noise(size, mu, sigma):
    """Return an array containing statistical noise with
    a Gaussian (normal) distribution with mean centred
    at given value.

    Parameters
    ----------
    size : size of the array
    mu : mean value of the Gaussian distribution
    sigma : standard deviation of the Gaussian distribution

    Returns
    -------
    noise : array containing the noise
    """
    return np.array([random.gauss(mu, sigma) for _ in range(size)], dtype=float)


def cubic_Bezier(p0, p1, p2, p3, n, linear=True):
    """Generate a cubic Bezier curve.

    Parameters
    ----------
    p0 : starting control point
    p1 : first anchor control point
    p2 : second anchor control point
    p3 : last control point
    n : number of points in the curve
    linear : return linearly sampled Bezier curve

    Returns
    -------
    x, y : arrays with the x and y components of Bezier curve

    Notes
    -----
    This function makes use of NumPy linear interpolation only.
    """
    t = np.linspace(0.0, 1.0, n)
    x = (
        (1 - t) ** 3 * p0[0]
        + t * p1[0] * (3 * (1 - t) ** 2)
        + p2[0] * (3 * (1 - t) * t**2)
        + p3[0] * t**3
    )
    y = (
        (1 - t) ** 3 * p0[1]
        + t * p1[1] * (3 * (1 - t) ** 2)
        + p2[1] * (3 * (1 - t) * t**2)
        + p3[1] * t**3
    )
    if linear:
        x_linear = np.linspace(0.0, 1.0, n)
        y_linear = np.interp(x_linear, x, y)
        return x_linear, y_linear
    return x, y


def find_closest(array, value):
    """
    Find the index of the element in 'array' closest to 'value'.

    Parameters
    ----------
    array : array-like
        Input array (must be 1D).
    value : float
        Target value.

    Returns
    -------
    idx : int
        Index of the closest element in the array.
    """
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
    return idx


############################## Plotting helpers ##############################


def _line_plot(path, title, xlabel, ylabel, series, legend=True):
    """
    Save a line plot. `series` is a list of (x, y, fmt, kwargs) tuples,
    passed to plt.plot.
    """
    plt.figure(figsize=(10, 5))
    for x, y, fmt, kwargs in series:
        plt.plot(x, y, fmt, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend()
    plt.grid()
    plt.savefig(path, dpi=150)
    plt.close()


def _bar_plot(path, title, xlabel, ylabel, values, color):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(values)), values, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.savefig(path, dpi=150)
    plt.close()


def _heatmap(path, title, xlabel, ylabel, data, extent, cmap, colorbar_label):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect="auto", origin="lower", extent=extent, cmap=cmap)
    plt.colorbar(label=colorbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, dpi=150)
    plt.close()


############################## Bandpass Amp ##############################################


def bandpass_amplitude(freqs_mhz, spline_data, plot_output_dir: Optional[str] = None):
    """
    Calculate amplitude values for bandpass, from B-spline fits to the
    measured bandpass. Plots are saved if plot_output_dir is given.

    Returns
    -------
    dict of np.ndarray of shape (n_channels,), keyed by polarisation "X", "Y"
    """
    spline_frequencies = spline_data["frequencies"]
    amplitudes = {}
    for pol in ("X", "Y"):
        spline = BSpline(
            spline_data[f"knots_{pol}"],
            spline_data[f"coefficients_{pol}"],
            spline_data[f"degree_{pol}"],
        )
        amplitudes[pol] = spline(freqs_mhz)

        if plot_output_dir:
            measured = spline_data[
                f"Perturbed_Vogel_HARP_station_pol{pol}_Az00ZA00_S21_LNA_norm"
            ]
            # Evaluate splines over all frequencies for comparison
            _line_plot(
                f"{plot_output_dir}/bspline_bandpass_amp_{pol}.png",
                f"Bandpass amplitude — {pol} polarisation",
                "Frequency (MHz)",
                "Amplitude (arbitrary units)",
                [
                    (
                        spline_frequencies,
                        measured,
                        "o",
                        {"label": f"Measured {pol}-pol", "alpha": 0.5},
                    ),
                    (
                        spline_frequencies,
                        spline(spline_frequencies),
                        "-",
                        {"label": f"Spline fit {pol}-pol"},
                    ),
                    (
                        freqs_mhz,
                        amplitudes[pol],
                        ".",
                        {"label": f"Simulation {pol}-pol"},
                    ),
                ],
            )
    return amplitudes


############################## Bandpass Phase ##############################################


def bandpass_phase(axes: GainAxes, spline_data, plot_output_dir: Optional[str] = None):
    """
    Calculate phase values (degrees) for bandpass, from Bezier curves.
    Plots are saved if plot_output_dir is given.

    Returns
    -------
    dict of np.ndarray of shape (n_channels,), keyed by polarisation "X", "Y"
    """
    spline_frequencies = spline_data["frequencies"]
    # The simulated band must start and end on the spline frequency grid
    start_idx = np.flatnonzero(spline_frequencies == axes.start_freq_mhz)
    end_idx = np.flatnonzero(spline_frequencies == axes.end_freq_mhz)
    if not (start_idx.size and end_idx.size):
        raise ValueError(
            "simulation_start_frequency_hz and simulation_end_frequency_hz "
            "must be on the frequency grid of the bandpass spline file: "
            f"{spline_frequencies[0]:g} to {spline_frequencies[-1]:g} MHz, in "
            f"steps of {spline_frequencies[1] - spline_frequencies[0]:g} MHz. "
            f"Got {axes.start_freq_mhz:g} to {axes.end_freq_mhz:g} MHz"
        )
    band = slice(start_idx[0], end_idx[0] + 1)
    phases = {}
    for pol, control_points in BANDPASS_PHASE_CONTROL_POINTS.items():
        _, bezier_phase = cubic_Bezier(
            *control_points, n=len(spline_frequencies), linear=True
        )
        # Create phase profile used for simulation.
        phases[pol] = np.interp(
            axes.freqs_mhz, spline_frequencies[band], bezier_phase[band]
        )

        if plot_output_dir:
            _line_plot(
                f"{plot_output_dir}/bazier_bandpass_phase_{pol}.png",
                f"Bandpass phase — {pol} polarisation",
                "Frequency (MHz)",
                "Phase (degrees)",
                [
                    (
                        spline_frequencies,
                        bezier_phase,
                        "-",
                        {"label": f"Bezier fit {pol}-pol"},
                    ),
                    (
                        axes.freqs_mhz,
                        phases[pol],
                        ".",
                        {"label": f"Simulation {pol}-pol"},
                    ),
                ],
            )
    return phases


############################## Gaussian noise per station (both amp and phase) ##############################


def bandpass_offset_per_station(n_stations, plot_output_dir: Optional[str] = None):
    """
    Random amplitude and phase (degrees) offsets, added to each station
    bandpass.

    Returns
    -------
    amplitude_offset_per_station, phase_offset_per_station
        np.ndarray of shape (n_stations,)
    """
    amplitude_offset_per_station = create_gaussian_noise(n_stations, 0.0, 0.08)
    phase_offset_per_station = create_gaussian_noise(n_stations, -2.0, 2.0)

    if plot_output_dir:
        _bar_plot(
            f"{plot_output_dir}/amplitude_offset_stations.png",
            "Amplitude offset per station",
            "Station index",
            "Amplitude offset",
            amplitude_offset_per_station,
            "skyblue",
        )
        _bar_plot(
            f"{plot_output_dir}/phase_offset_stations.png",
            "Phase offset per station",
            "Station index",
            "Phase offset (degrees)",
            phase_offset_per_station,
            "salmon",
        )

    return amplitude_offset_per_station, phase_offset_per_station


############################## Time dependent effects (both amp and phase) ##############################


def time_variant_effects(axes: GainAxes, plot_output_dir: Optional[str] = None):
    """
    Calculates time variant effects added to the bandpass: a sinusoid per
    station, with a random period and phase.

    Returns
    -------
    amplitude_time_variation_profile, phase_time_variation_profile
        np.ndarrays[float] of shape (time_samples, n_stations)
    """
    n_stations = axes.n_stations
    # The order of the random draws is kept, for repeatability.
    amplitude_variation_frequency = 1 / create_gaussian_noise(
        n_stations, axes.n_times * 0.75, 100
    )  # 0.75 is the number of scans in ~2 hours with 100 samples sigma.
    phase_variation_frequency = 1 / create_gaussian_noise(
        n_stations, axes.n_times * 0.5, 30
    )  # 0.5 is the number of scans in 3000 seconds with 30 samples sigma.
    amplitude_function_phase = create_gaussian_noise(n_stations, 0, 0.1)  # radians
    phase_function_phase = create_gaussian_noise(n_stations, 0, 0.2)  # radians

    def sinusoid(offset, amplitude, frequency, phase):
        return offset + amplitude * np.sin(
            2 * np.pi * frequency * axes.times_s[:, np.newaxis] + phase
        )

    amplitude_time_variation_profile = sinusoid(
        1, 0.1, amplitude_variation_frequency, amplitude_function_phase
    )
    phase_time_variation_profile = sinusoid(
        0, 0.5, phase_variation_frequency, phase_function_phase
    )

    if plot_output_dir:
        times = axes.times_s
        first_stations = range(min(3, n_stations))
        extent = [times[0], times[-1], 0, n_stations]
        for name, profile, ylabel, cmap, colorbar_label in [
            (
                "amplitude",
                amplitude_time_variation_profile,
                "Amplitude",
                "viridis",
                "Amplitude factor",
            ),
            (
                "phase",
                phase_time_variation_profile,
                "Phase (degrees)",
                "twilight",
                "Phase (degrees)",
            ),
        ]:
            # ---- 1. Plot single-station time series ----
            _line_plot(
                f"{plot_output_dir}/{name}_time_series.png",
                f"{name.capitalize()} variation vs time (first 3 stations)",
                "Time (seconds)",
                ylabel,
                [
                    (times, profile[:, st], "-", {"label": f"Station {st}"})
                    for st in first_stations
                ],
            )
            # ---- 2. Plot heatmaps for all stations ----
            _heatmap(
                f"{plot_output_dir}/{name}_variation_heatmap.png",
                f"{name.capitalize()} time variation profile (all stations)",
                "Time (seconds)",
                "Station index",
                profile.T,
                extent,
                cmap,
                colorbar_label,
            )

    return amplitude_time_variation_profile, phase_time_variation_profile


############################## Additional RFI ##############################


def calculate_rfi(
    axes: GainAxes,
    rfi_start_freq,
    rfi_end_freq,
    plot_output_dir: Optional[str] = None,
):
    """
    Generate RFI to inject into the gain table. Frequencies are in MHz.

    Returns
    -------
    np.complex128, slice
        Complex RFI values with shape (time, channel, station, pol),
        python array slice in frequency
    """
    rfi_start_freq_idx = find_closest(axes.freqs_mhz, rfi_start_freq)
    rfi_end_freq_idx = find_closest(axes.freqs_mhz, rfi_end_freq)
    RFI_n_channels = rfi_end_freq_idx - rfi_start_freq_idx

    RFI_amplitude_max = 10
    RFI_amplitude_min = 2
    RFI_phase_max = 20
    RFI_phase_min = -20
    shape = (axes.n_times, RFI_n_channels, axes.n_stations, N_POLS)
    RFI_amplitude = np.zeros(shape, dtype=float)
    RFI_phase = np.zeros(shape, dtype=float)
    # The order of the random draws is kept, for repeatability.
    for i in range(axes.n_stations):
        for pol in range(N_POLS):
            RFI_amplitude[:, :, i, pol] = (
                RFI_amplitude_min - RFI_amplitude_max
            ) * np.random.random_sample(shape[:2]) + RFI_amplitude_max
            RFI_phase[:, :, i, pol] = (
                RFI_phase_min - RFI_phase_max
            ) * np.random.random_sample(shape[:2]) + RFI_phase_max
    RFI_complex = RFI_amplitude * np.exp(1j * np.deg2rad(RFI_phase))

    if plot_output_dir:
        plot_rfi(
            axes,
            RFI_amplitude,
            RFI_phase,
            rfi_start_freq,
            rfi_end_freq,
            plot_output_dir,
        )

    return RFI_complex, slice(rfi_start_freq_idx, rfi_end_freq_idx, None)


def plot_rfi(
    axes, RFI_amplitude, RFI_phase, rfi_start_freq, rfi_end_freq, plot_output_dir
):
    RFI_n_channels = RFI_amplitude.shape[1]
    times = np.arange(axes.n_times) * axes.sampling_time_s  # s (integration length)
    freqs = np.linspace(rfi_start_freq, rfi_end_freq, RFI_n_channels)  # MHz
    t_idx, st_idx, pol_idx = 0, 0, 0

    # --- 1. Spectrum at one time, one station, one pol ---
    _line_plot(
        f"{plot_output_dir}/rfi_amp_spectrum.png",
        f"RFI amplitude (time={t_idx}, station={st_idx}, pol={'X' if pol_idx==0 else 'Y'})",
        "Frequency (MHz)",
        "Amplitude",
        [
            (
                freqs,
                RFI_amplitude[t_idx, :, st_idx, pol_idx],
                "-",
                {"label": "Amplitude"},
            )
        ],
        legend=False,
    )
    _line_plot(
        f"{plot_output_dir}/rfi_phase_spectrum.png",
        f"RFI phase (time={t_idx}, station={st_idx}, pol={pol_idx})",
        "Frequency (MHz)",
        "Phase (degrees)",
        [(freqs, RFI_phase[t_idx, :, st_idx, pol_idx], "-", {"label": "Phase"})],
        legend=False,
    )

    # --- 2. Time series at one frequency, one station/pol ---
    f_idx = RFI_n_channels // 2
    _line_plot(
        f"{plot_output_dir}/rfi_amp_time_series.png",
        f"RFI amplitude vs time (freq_idx={freqs[f_idx]:.1f} MHz, station={st_idx}, pol={pol_idx})",
        "Time (s)",
        "Amplitude",
        [(np.arange(axes.n_times), RFI_amplitude[:, f_idx, st_idx, pol_idx], "-", {})],
        legend=False,
    )

    # --- 3. Heatmap (freq × time) for one station/pol ---
    _heatmap(
        f"{plot_output_dir}/rfi_amp_heatmap.png",
        f"RFI amplitude heatmap (station={st_idx}, pol={pol_idx})",
        "Time (s)",
        "Frequency (MHz)",
        RFI_amplitude[:, :, st_idx, pol_idx].T,
        [times[0], times[-1], freqs[0], freqs[-1]],
        "inferno",
        "Amplitude",
    )

    print(
        "Saved: rfi_amp_spectrum.png, rfi_phase_spectrum.png, rfi_amp_time_series.png, rfi_amp_heatmap.png"
    )


############################## Add outliers to gains ##############################


def add_gain_outliers(gain, amp_range, n_stations_to_corrupt, n_channels_to_corrupt):
    """
    Add complex gain outliers by corrupting random stations and channels.

    Parameters
    ----------
    gain : np.ndarray
        Complex gain array with shape (n_time, n_channels, n_stations)
    amp_range: tuple(int, int)
        Range of outlier amplitude
    n_stations_to_corrupt : int
        Number of random stations to corrupt
    n_channels_to_corrupt : int
        Number of random channels per station to corrupt

    Returns
    -------
    gain_with_outliers : np.ndarray
        Complex gain array with outliers injected
    """

    gain = gain.copy()

    _, n_channels, n_stations = gain.shape

    corrupt_stations = np.random.choice(
        n_stations, size=n_stations_to_corrupt, replace=False
    )

    for st in corrupt_stations:
        corrupt_channels = np.random.choice(
            n_channels, size=n_channels_to_corrupt, replace=False
        )

        for ch in corrupt_channels:
            amp_multiplier = np.random.uniform(*amp_range)
            phase_offset = np.random.uniform(-180.0, 180.0)

            gain[:, ch, st] *= amp_multiplier * np.exp(1j * np.deg2rad(phase_offset))

    return gain


############################## Generate gaintables combining all effects ##############################


def combine_gain_effects(
    bandpass_amp, bandpass_phase_deg, amp_time, phase_time, amp_offset, phase_offset
):
    """
    Combine the bandpass (per channel), time variations (per time and
    station) and station offsets (per station) of one polarisation.

    Returns
    -------
    np.ndarray[complex] of shape (n_time, n_channels, n_stations)
    """
    gain = (
        amp_time
        * (
            bandpass_amp[:, np.newaxis, np.newaxis]
            + amp_offset[np.newaxis, np.newaxis, :]
        )
    ) * np.exp(
        1j
        * np.deg2rad(
            phase_time[np.newaxis, :, :]
            + bandpass_phase_deg[:, np.newaxis, np.newaxis]
            + phase_offset[np.newaxis, np.newaxis, :]
        )
    )
    return np.swapaxes(gain, 0, 1)


def calculate_gains(cfg: SimConfig):
    """
    Generate the gains of both polarisations.

    Returns
    -------
    gain_xpol, gain_ypol, frequencies (MHz)
        gains with shape (n_time, n_channels, n_stations)
    """
    start_time = time.perf_counter()
    seed_random_generators()

    gt_cfg = cfg.require_generate_gaintable()
    axes = GainAxes.from_config(cfg)

    plot_output_dir = gt_cfg.plot_output_dir if gt_cfg.plot else None
    if plot_output_dir:
        os.makedirs(plot_output_dir, exist_ok=True)

    spline_data = np.load(gt_cfg.bandpass_spline_file)

    amplitudes = bandpass_amplitude(axes.freqs_mhz, spline_data, plot_output_dir)
    phases = bandpass_phase(axes, spline_data, plot_output_dir)

    # The random draws must happen in this order: time variations, station
    # offsets, outliers (X then Y), RFI.
    if gt_cfg.time_variant:
        amp_time, phase_time = time_variant_effects(axes, plot_output_dir)
    else:
        amp_time = np.ones((axes.n_times, axes.n_stations), dtype=np.float64)
        phase_time = np.zeros((axes.n_times, axes.n_stations), dtype=np.float64)

    if gt_cfg.station_offset:
        amp_offset, phase_offset = bandpass_offset_per_station(
            axes.n_stations, plot_output_dir
        )
    else:
        amp_offset = phase_offset = np.zeros(axes.n_stations)

    gains = {
        pol: combine_gain_effects(
            amplitudes[pol], phases[pol], amp_time, phase_time, amp_offset, phase_offset
        )
        for pol in ("X", "Y")
    }

    outliers = gt_cfg.outlier_config
    if outliers.enable:
        n_channels = len(axes.freqs_mhz)
        if outliers.n_channels_to_corrupt > n_channels:
            raise ValueError(
                "generate_gaintable.outlier_config.n_channels_to_corrupt "
                f"({outliers.n_channels_to_corrupt}) must be <= the number of "
                f"channels ({n_channels})"
            )
        for pol in ("X", "Y"):
            gains[pol] = add_gain_outliers(
                gains[pol],
                (outliers.amp_min, outliers.amp_max),
                outliers.n_stations_to_corrupt,
                outliers.n_channels_to_corrupt,
            )

    if gt_cfg.rfi:
        RFI_complex, freq_slice = calculate_rfi(
            axes,
            gt_cfg.rfi_start_freq_hz * 1e-6,
            gt_cfg.rfi_end_freq_hz * 1e-6,
            plot_output_dir,
        )
        # Inject RFI into gain tables.
        for pol_idx, pol in enumerate(("X", "Y")):
            gains[pol][:, freq_slice, :] = RFI_complex[:, :, :, pol_idx]

    end_time = time.perf_counter()
    print("Total processing time: " + str(int(np.ceil(end_time - start_time))) + " s.")

    return gains["X"], gains["Y"], axes.freqs_mhz


def write_gaintable(path, freqs_hz, gain_xpol, gain_ypol):
    """Write the gains in the format of an OSKAR gain model (gain_model.h5)."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("freq (Hz)", data=freqs_hz)
        f.create_dataset("gain_xpol", data=gain_xpol)
        f.create_dataset("gain_ypol", data=gain_ypol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gt_cfg = cfg.require_generate_gaintable()
    if not os.path.isfile(gt_cfg.bandpass_spline_file):
        raise FileNotFoundError(
            "generate_gaintable.bandpass_spline_file not found: "
            f"{gt_cfg.bandpass_spline_file}"
        )
    output_gaintable_path = gt_cfg.output_gaintable

    gain_xpol, gain_ypol, sim_freqs = calculate_gains(cfg)
    write_gaintable(output_gaintable_path, sim_freqs * 1e6, gain_xpol, gain_ypol)

    print(f"Wrote gains to {output_gaintable_path}")


if __name__ == "__main__":
    main()
