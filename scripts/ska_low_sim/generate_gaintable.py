#!/usr/bin/env python3
"""
Gain table generation script for radio interferometry simulations.

Authored by:
- Maciej Serylak
- Team Dhruva

The following effects can be simulated:
- Bandpass amplitude shape across frequency, fitted to
  measured data and perturbed across stations.
- Bandpass phase slopes across frequency (bazier curves)
- Slow time-varying gain and phase fluctuations, simulated as sinusoidal
  functions (e.g. amplitude variations with ~2.4 h period, phase variations
  with ~0.9 h period).
- Random per-station offsets in amplitude and phase.
- RFI (radio-frequency interference), injected as a large amplitude
  perturbation within a specified frequency band.

The script needs a YAML config file as input, which contains the necessary
parameters.

The generated gain solutions are written to an H5parm file with datasets:
- "freq (Hz)"   : frequency axis in Hz
- "gain_xpol"   : complex gain solutions for X polarization
- "gain_ypol"   : complex gain solutions for Y polarization

Usage:
python generate_gaintable.py sim.yaml
"""

import random

import yaml

import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

# Setting seed to a fixed value in order to achieve repeatability of results.
random.seed(100)
np.random.seed(100)

SPLINE_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "SKA_Low_AA2_SP5175_spline_data.npz"
)


def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


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
    noise = np.zeros(size)
    for i in range(size):
        noise[i] = random.gauss(mu, sigma)
    return noise


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
    return (x_linear, y_linear) if linear else (x, y)


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


############################## Bandpass Amp ##############################################


def bandpass_amplitude(
    simulation_frequency_table, spline_data, plot=False, plot_output_dir="."
):
    """
    Calculate phase values for bandpass

    Returns
    -------
    station_bandpass_amplitude_X, station_bandpass_amplitude_Y
        np.ndarray of shape (n_channels,)
    """
    spline_frequencies = spline_data["frequencies"]
    # Load coefficients from B-spline fitting to create bandpass that will be used for creating the gain table.
    Perturbed_Vogel_HARP_station_polX_Az00ZA00_S21_LNA_norm = spline_data[
        "Perturbed_Vogel_HARP_station_polX_Az00ZA00_S21_LNA_norm"
    ]
    knots_X = spline_data["knots_X"]
    coefficients_X = spline_data["coefficients_X"]
    degree_X = spline_data["degree_X"]
    Perturbed_Vogel_HARP_station_polY_Az00ZA00_S21_LNA_norm = spline_data[
        "Perturbed_Vogel_HARP_station_polY_Az00ZA00_S21_LNA_norm"
    ]
    knots_Y = spline_data["knots_Y"]
    coefficients_Y = spline_data["coefficients_Y"]
    degree_Y = spline_data["degree_Y"]

    spline_X = BSpline(knots_X, coefficients_X, degree_X)
    spline_Y = BSpline(knots_Y, coefficients_Y, degree_Y)
    station_bandpass_amplitude_X = spline_X(simulation_frequency_table)
    station_bandpass_amplitude_Y = spline_Y(simulation_frequency_table)

    if plot:
        # Evaluate splines over all frequencies for comparision
        amp_X_fit = spline_X(spline_frequencies)
        amp_Y_fit = spline_Y(spline_frequencies)

        plt.figure(figsize=(10, 5))
        plt.plot(
            spline_frequencies,
            Perturbed_Vogel_HARP_station_polX_Az00ZA00_S21_LNA_norm,
            "o",
            label="Measured X-pol",
            alpha=0.5,
        )
        plt.plot(spline_frequencies, amp_X_fit, "-", label="Spline fit X-pol")
        plt.plot(
            simulation_frequency_table,
            station_bandpass_amplitude_X,
            ".",
            label="Simulation X-pol",
        )
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (arbitrary units)")
        plt.title("Bandpass amplitude — X polarisation")
        plt.legend()
        plt.grid()
        plt.savefig(f"{plot_output_dir}/bspline_bandpass_amp_X.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(
            spline_frequencies,
            Perturbed_Vogel_HARP_station_polY_Az00ZA00_S21_LNA_norm,
            "o",
            label="Measured Y-pol",
            alpha=0.5,
        )
        plt.plot(spline_frequencies, amp_Y_fit, "-", label="Spline fit Y-pol")
        plt.plot(
            simulation_frequency_table,
            station_bandpass_amplitude_Y,
            ".",
            label="Simulation Y-pol",
        )
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (arbitrary units)")
        plt.title("Bandpass amplitude — Y polarisation")
        plt.legend()
        plt.grid()
        plt.savefig(f"{plot_output_dir}/bspline_bandpass_amp_Y.png", dpi=150)
        plt.close()

    return station_bandpass_amplitude_X, station_bandpass_amplitude_Y


############################## Bandpass Phase ##############################################


def bandpass_phase(
    simulation_start_frequency,
    simulation_end_frequency,
    simulation_frequency_table,
    spline_data,
    plot=False,
    plot_output_dir=".",
):
    """
    calculate phase values for bandpass

    Returns
    -------
    station_bandpass_phase_X, station_bandpass_phase_Y
        np.ndarray of shape (n_channels,)
    """
    spline_frequencies = spline_data["frequencies"]
    # Generate phases to be included in the station bandpass.
    p0_X = [0, 6]  # Define arbitrary control points for creating Bezier curve.
    p1_X = [0.2, -1]
    p2_X = [0.05, -3]
    p3_X = [1, -5]
    p0_Y = [0, 6.1]
    p1_Y = [0.1, -1.2]
    p2_Y = [0.05, -3.2]
    p3_Y = [1, -5.3]
    _, phase_X = cubic_Bezier(
        p0_X, p1_X, p2_X, p3_X, n=len(spline_frequencies), linear=True
    )
    _, phase_Y = cubic_Bezier(
        p0_Y, p1_Y, p2_Y, p3_Y, n=len(spline_frequencies), linear=True
    )

    # Create phase profile used for simulation.
    station_bandpass_phase_X = np.interp(
        simulation_frequency_table,
        spline_frequencies[
            np.where(spline_frequencies == simulation_start_frequency)[0][0] : np.where(
                spline_frequencies == simulation_end_frequency
            )[0][0]
            + 1
        ],
        phase_X[
            np.where(spline_frequencies == simulation_start_frequency)[0][0] : np.where(
                spline_frequencies == simulation_end_frequency
            )[0][0]
            + 1
        ],
    )
    station_bandpass_phase_Y = np.interp(
        simulation_frequency_table,
        spline_frequencies[
            np.where(spline_frequencies == simulation_start_frequency)[0][0] : np.where(
                spline_frequencies == simulation_end_frequency
            )[0][0]
            + 1
        ],
        phase_Y[
            np.where(spline_frequencies == simulation_start_frequency)[0][0] : np.where(
                spline_frequencies == simulation_end_frequency
            )[0][0]
            + 1
        ],
    )

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(spline_frequencies, phase_X, "-", label="Bazier fit X-pol")
        plt.plot(
            simulation_frequency_table,
            station_bandpass_phase_X,
            ".",
            label="Simulation X-pol",
        )
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Phase (degrees)")
        plt.title("Bandpass phase — X polarisation")
        plt.legend()
        plt.grid()
        plt.savefig(f"{plot_output_dir}/bazier_bandpass_phase_X.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(spline_frequencies, phase_Y, "-", label="Bazier fit Y-pol")
        plt.plot(
            simulation_frequency_table,
            station_bandpass_phase_Y,
            ".",
            label="Simulation Y-pol",
        )
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Phase (degrees)")
        plt.title("Bandpass phase — Y polarisation")
        plt.legend()
        plt.grid()
        plt.savefig(f"{plot_output_dir}/bazier_bandpass_phase_Y.png", dpi=150)
        plt.close()

    return station_bandpass_phase_X, station_bandpass_phase_Y


############################## Gaussian noise per station (both amp and phase) ##############################################


def bandpass_offset_per_station(n_stations, plot=False, plot_output_dir="."):
    """

    Returns
    -------
    amplitude_offset_per_station, phase_offset_per_station
        np.ndarray of shape (n_stations,)
    """
    # Create an array of offset values that will be added to each station bandpass
    mu = 0.0
    sigma = 0.08
    amplitude_offset_per_station = create_gaussian_noise(n_stations, mu, sigma)
    # Create an array of offset values that will be added to each station phase.
    mu = -2.0
    sigma = 2.0
    phase_offset_per_station = create_gaussian_noise(n_stations, mu, sigma)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(n_stations), amplitude_offset_per_station, color="skyblue")
        plt.xlabel("Station index")
        plt.ylabel("Amplitude offset")
        plt.title("Amplitude offset per station")
        plt.grid(axis="y")
        plt.savefig(f"{plot_output_dir}/amplitude_offset_stations.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(n_stations), phase_offset_per_station, color="salmon")
        plt.xlabel("Station index")
        plt.ylabel("Phase offset (degrees)")
        plt.title("Phase offset per station")
        plt.grid(axis="y")
        plt.savefig(f"{plot_output_dir}/phase_offset_stations.png", dpi=150)
        plt.close()

    return amplitude_offset_per_station, phase_offset_per_station


############################## Time dependent effects (both amp and phase) ##############################


def time_variant_effects(
    calibration_time,
    n_stations,
    number_of_cal_time_samples,
    plot=False,
    plot_output_dir=".",
):
    """
    Calculates time variatnt effects added to the bandpass

    Returns
    -------
    amplitude_time_variation_profile, phase_time_variation_profile
        np.ndarrays[float] of shape (time_samples, n_stations)
    """
    # Create a table of amplitude and phase variations.
    amplitude_time_variation_profile = np.zeros(
        (len(calibration_time), n_stations), dtype=np.float64
    )
    phase_time_variation_profile = np.zeros(
        (len(calibration_time), n_stations), dtype=np.float64
    )
    amplitude_variation_frequency = 1 / create_gaussian_noise(
        n_stations, number_of_cal_time_samples * 0.75, 100
    )  # 0.75 is the number of scans in ~2 hours with 100 samples sigma.
    phase_variation_frequency = 1 / create_gaussian_noise(
        n_stations, number_of_cal_time_samples * 0.5, 30
    )  # 0.5 is the number of scans in 3000 seconds with 30 samples sigma.

    amplitude_fuction_amplitude = 0.1  # Amplitude of function.
    amplitude_fuction_phase = create_gaussian_noise(
        n_stations, 0, 0.1
    )  # Phase of function in radians.
    amplitude_fuction_offset = 1  # Offset of function.
    phase_fuction_amplitude = 0.5  # Amplitude of function.
    phase_fuction_phase = create_gaussian_noise(
        n_stations, 0, 0.2
    )  # Phase of function in radians.
    phase_fuction_offset = 0  # Offset of function.

    # Prepare bandpass amplitude and phase variation profiles.
    amplitude_time_variation_profile = amplitude_fuction_offset + (
        amplitude_fuction_amplitude
        * np.sin(
            2 * np.pi * amplitude_variation_frequency * calibration_time[:, np.newaxis]
            + amplitude_fuction_phase
        )
    )
    phase_time_variation_profile = phase_fuction_offset + (
        phase_fuction_amplitude
        * np.sin(
            2 * np.pi * phase_variation_frequency * calibration_time[:, np.newaxis]
            + phase_fuction_phase
        )
    )

    if plot:
        # ---- 1. Plot single-station time series ----
        plt.figure(figsize=(10, 5))
        for st in range(min(3, n_stations)):  # show first 3 stations
            plt.plot(
                calibration_time,
                amplitude_time_variation_profile[:, st],
                label=f"Station {st}",
            )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Amplitude variation vs time (first 3 stations)")
        plt.legend()
        plt.grid()
        plt.savefig(f"{plot_output_dir}/amplitude_time_series.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        for st in range(min(3, n_stations)):
            plt.plot(
                calibration_time,
                (phase_time_variation_profile[:, st]),
                label=f"Station {st}",
            )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Phase (degrees)")
        plt.title("Phase variation vs time (first 3 stations)")
        plt.legend()
        plt.grid()
        plt.savefig(f"{plot_output_dir}/phase_time_series.png", dpi=150)
        plt.close()

        # ---- 2. Plot heatmaps for all stations ----
        plt.figure(figsize=(10, 6))
        plt.imshow(
            amplitude_time_variation_profile.T,
            aspect="auto",
            origin="lower",
            extent=[calibration_time[0], calibration_time[-1], 0, n_stations],
            cmap="viridis",
        )
        plt.colorbar(label="Amplitude factor")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Station index")
        plt.title("Amplitude time variation profile (all stations)")
        plt.savefig(f"{plot_output_dir}/amplitude_variation_heatmap.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.imshow(
            (phase_time_variation_profile.T),
            aspect="auto",
            origin="lower",
            extent=[calibration_time[0], calibration_time[-1], 0, n_stations],
            cmap="twilight",
        )
        plt.colorbar(label="Phase (degrees)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Station index")
        plt.title("Phase time variation profile (all stations)")
        plt.savefig(f"{plot_output_dir}/phase_variation_heatmap.png", dpi=150)
        plt.close()

    return amplitude_time_variation_profile, phase_time_variation_profile


############################## Additional RFI ##############################


def calculate_rfi(
    simulation_frequency_table,
    rfi_start_freq,
    rfi_end_freq,
    number_of_cal_time_samples,
    sampling_time,
    n_stations,
    n_pols,
    plot=False,
    plot_output_dir=".",
):
    """
    Generate RFI to inject into the gain table.

    Returns
    -------
    np.complex128, slice
        Complex RFI values, python array slice in frequency
    """
    rfi_start_freq_idx = find_closest(simulation_frequency_table, rfi_start_freq)
    rfi_end_freq_idx = find_closest(simulation_frequency_table, rfi_end_freq)
    RFI_n_channels = rfi_end_freq_idx - rfi_start_freq_idx

    RFI_amplitude_max = 10
    RFI_amplitude_min = 2
    RFI_amplitude = np.zeros(
        (number_of_cal_time_samples, RFI_n_channels, n_stations, n_pols),
        dtype=float,
    )
    RFI_phase_max = 20
    RFI_phase_min = -20
    RFI_phase = np.zeros(
        (number_of_cal_time_samples, RFI_n_channels, n_stations, n_pols),
        dtype=float,
    )
    for i in range(n_stations):
        for l in range(n_pols):
            RFI_amplitude[:, :, i, l] = (
                RFI_amplitude_min - RFI_amplitude_max
            ) * np.random.random_sample(
                (number_of_cal_time_samples, RFI_n_channels)
            ) + RFI_amplitude_max
            RFI_phase[:, :, i, l] = (
                RFI_phase_min - RFI_phase_max
            ) * np.random.random_sample(
                (number_of_cal_time_samples, RFI_n_channels)
            ) + RFI_phase_max
    RFI_complex = RFI_amplitude * np.exp(1j * np.deg2rad(RFI_phase))

    if plot:
        times = (
            np.arange(number_of_cal_time_samples) * sampling_time
        )  # s (integration length)
        freqs = np.linspace(rfi_start_freq, rfi_end_freq, RFI_n_channels)  # MHz
        # --- 1. Spectrum at one time, one station, one pol ---
        t_idx, st_idx, pol_idx = 0, 0, 0
        plt.figure(figsize=(10, 5))
        plt.plot(freqs, RFI_amplitude[t_idx, :, st_idx, pol_idx], label="Amplitude")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude")
        plt.title(
            f"RFI amplitude (time={t_idx}, station={st_idx}, pol={'X' if pol_idx==0 else 'Y'})"
        )
        plt.grid()
        plt.savefig(f"{plot_output_dir}/rfi_amp_spectrum.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(freqs, RFI_phase[t_idx, :, st_idx, pol_idx], label="Phase")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Phase (degrees)")
        plt.title(f"RFI phase (time={t_idx}, station={st_idx}, pol={pol_idx})")
        plt.grid()
        plt.savefig(f"{plot_output_dir}/rfi_phase_spectrum.png", dpi=150)
        plt.close()

        # --- 2. Time series at one frequency, one station/pol ---
        f_idx = RFI_n_channels // 2
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(number_of_cal_time_samples),
            RFI_amplitude[:, f_idx, st_idx, pol_idx],
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(
            f"RFI amplitude vs time (freq_idx={freqs[f_idx]:.1f} MHz, station={st_idx}, pol={pol_idx})"
        )
        plt.grid()
        plt.savefig(f"{plot_output_dir}/rfi_amp_time_series.png", dpi=150)
        plt.close()

        # --- 3. Heatmap (freq × time) for one station/pol ---

        plt.figure(figsize=(10, 6))
        plt.imshow(
            RFI_amplitude[:, :, st_idx, pol_idx].T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            cmap="inferno",
        )
        plt.colorbar(label="Amplitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.title(f"RFI amplitude heatmap (station={st_idx}, pol={pol_idx})")
        plt.savefig(f"{plot_output_dir}/rfi_amp_heatmap.png", dpi=150)
        plt.close()

        print(
            "Saved: rfi_amp_spectrum.png, rfi_phase_spectrum.png, rfi_amp_time_series.png, rfi_amp_heatmap.png"
        )

    return RFI_complex, slice(rfi_start_freq_idx, rfi_end_freq_idx, None)


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

    corrupt_stations = np.random.choice(n_stations, size=n_stations_to_corrupt, replace=False)

    for st in corrupt_stations:
        corrupt_channels = np.random.choice(n_channels, size=n_channels_to_corrupt, replace=False)
        
        for ch in corrupt_channels:
            amp_multiplier = np.random.uniform(*amp_range)
            phase_offset = np.random.uniform(-180.0, 180.0)
            
            gain[:, ch, st] *= amp_multiplier * np.exp(1j * np.deg2rad(phase_offset))

    return gain


############################## Generate gaintables combining all effects ##############################


def calculate_gains(cfg):
    start_time = time.perf_counter()

    # -------------- Unpack parameters ---------------- #

    # Common simulation parameters
    n_stations = cfg["n_stations"]
    simulation_start_frequency = float(cfg["simulation_start_frequency_hz"]) * 1e-6
    simulation_end_frequency = float(cfg["simulation_end_frequency_hz"]) * 1e-6
    correlated_channel_bandwidth = float(cfg["correlated_channel_bandwidth_hz"]) * 1e-6
    observing_time_cal = float(cfg["observing_time_mins"]) * 60
    sampling_time = cfg["sampling_time_sec"]

    # generate_gaintable specific parameters
    generate_gaintable_cfg = cfg["generate_gaintable"]

    outlier_config = generate_gaintable_cfg["outlier_config"]
    outlier_enable = outlier_config["enable"]

    station_offset = generate_gaintable_cfg.get("station_offset", True)
    time_variant = generate_gaintable_cfg.get("time_variant", True)

    rfi = generate_gaintable_cfg.get("rfi", False)
    if rfi:
        rfi_start_freq = float(cfg["rfi_start_freq_hz"]) * 1e-6
        rfi_end_freq = float(cfg["rfi_end_freq_hz"]) * 1e-6
    else:
        rfi_start_freq = rfi_end_freq = None

    plot = generate_gaintable_cfg.get("plot", False)
    plot_output_dir = generate_gaintable_cfg.get(
        "plot_output_dir", "./gaintable_generation_plots"
    )
    if plot:
        os.makedirs(plot_output_dir, exist_ok=True)

    # ---------------- Setup ---------------------#
    n_pols = 2
    spline_data = np.load(SPLINE_DATA_PATH)

    AA2_bandwidth = simulation_end_frequency - simulation_start_frequency

    number_of_correlated_channels = int(AA2_bandwidth / correlated_channel_bandwidth)
    number_of_cal_time_samples = int(observing_time_cal // sampling_time)

    # TODO: Verify what is the correct way to call arange
    simulation_frequency_table = np.arange(
        simulation_start_frequency,
        simulation_end_frequency,
        correlated_channel_bandwidth,
    )
    # TODO: Verify what is the correct way to call arange
    calibration_time = np.arange(
        0, number_of_cal_time_samples * sampling_time, sampling_time
    )

    # ----------------- Start calculating gains ------------ #

    gain_xpol = np.zeros(
        (
            len(simulation_frequency_table),
            number_of_cal_time_samples,
            n_stations,
        ),
        dtype=complex,
    )
    gain_ypol = np.zeros(
        (
            len(simulation_frequency_table),
            number_of_cal_time_samples,
            n_stations,
        ),
        dtype=complex,
    )

    station_bandpass_amplitude_X, station_bandpass_amplitude_Y = bandpass_amplitude(
        simulation_frequency_table=simulation_frequency_table,
        spline_data=spline_data,
        plot=plot,
        plot_output_dir=plot_output_dir,
    )
    station_bandpass_phase_X, station_bandpass_phase_Y = bandpass_phase(
        simulation_start_frequency=simulation_start_frequency,
        simulation_end_frequency=simulation_end_frequency,
        simulation_frequency_table=simulation_frequency_table,
        spline_data=spline_data,
        plot=plot,
        plot_output_dir=plot_output_dir,
    )

    if time_variant:
        amplitude_time_variation_profile, phase_time_variation_profile = (
            time_variant_effects(
                calibration_time=calibration_time,
                n_stations=n_stations,
                number_of_cal_time_samples=number_of_cal_time_samples,
                plot=plot,
                plot_output_dir=plot_output_dir,
            )
        )
    else:
        amplitude_time_variation_profile = np.ones(
            (len(calibration_time), n_stations), dtype=np.float64
        )
        phase_time_variation_profile = np.zeros(
            (len(calibration_time), n_stations), dtype=np.float64
        )

    if station_offset:
        amplitude_offset_per_station, phase_offset_per_station = (
            bandpass_offset_per_station(
                n_stations=n_stations, plot=plot, plot_output_dir=plot_output_dir
            )
        )
    else:
        amplitude_offset_per_station = phase_offset_per_station = np.zeros(n_stations)

    gain_xpol = (
        amplitude_time_variation_profile
        * (
            station_bandpass_amplitude_X[:, np.newaxis, np.newaxis]
            + amplitude_offset_per_station[np.newaxis, np.newaxis, :]
        )
    ) * np.exp(
        1j
        * np.deg2rad(
            phase_time_variation_profile[np.newaxis, :, :]
            + station_bandpass_phase_X[:, np.newaxis, np.newaxis]
            + phase_offset_per_station[np.newaxis, np.newaxis, :]
        )
    )
    gain_ypol = (
        amplitude_time_variation_profile
        * (
            station_bandpass_amplitude_Y[:, np.newaxis, np.newaxis]
            + amplitude_offset_per_station[np.newaxis, np.newaxis, :]
        )
    ) * np.exp(
        1j
        * np.deg2rad(
            phase_time_variation_profile[np.newaxis, :, :]
            + station_bandpass_phase_Y[:, np.newaxis, np.newaxis]
            + phase_offset_per_station[np.newaxis, np.newaxis, :]
        )
    )
    gain_xpol = np.swapaxes(gain_xpol, 0, 1)
    gain_ypol = np.swapaxes(gain_ypol, 0, 1)

    if outlier_enable:
        amp_range = (outlier_config["amp_min"],outlier_config["amp_max"])
        gain_xpol = add_gain_outliers(
            gain_xpol,
            amp_range,
            outlier_config["n_stations_to_corrupt"],
            outlier_config["n_channels_to_corrupt"],
        )
        gain_ypol = add_gain_outliers(
            gain_ypol,
            amp_range,
            outlier_config["n_stations_to_corrupt"],
            outlier_config["n_channels_to_corrupt"],
        )

    if rfi:
        RFI_complex, freq_slice = calculate_rfi(
            simulation_frequency_table=simulation_frequency_table,
            rfi_start_freq=rfi_start_freq,
            rfi_end_freq=rfi_end_freq,
            number_of_cal_time_samples=number_of_cal_time_samples,
            sampling_time=sampling_time,
            n_stations=n_stations,
            n_pols=n_pols,
            plot=plot,
            plot_output_dir=plot_output_dir,
        )
        # Inject RFI into gain tables.
        gain_xpol[:, freq_slice, :] = RFI_complex[:, :, :, 0]
        gain_ypol[:, freq_slice, :] = RFI_complex[:, :, :, 1]

    end_time = time.perf_counter()
    print("Total processing time: " + str(int(np.ceil(end_time - start_time))) + " s.")

    return gain_xpol, gain_ypol, simulation_frequency_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to YAML config file")

    args = parser.parse_args()

    cfg = load_config(args.config)

    output_gaintable_path = cfg["generate_gaintable"]["output_gaintable"]
    os.makedirs(os.path.dirname(output_gaintable_path), exist_ok=True)

    gain_xpol, gain_ypol, sim_freqs = calculate_gains(cfg)

    with h5py.File(output_gaintable_path, "w") as f:
        f.create_dataset("freq (Hz)", data=sim_freqs * 1e6)
        f.create_dataset("gain_xpol", data=gain_xpol)
        f.create_dataset("gain_ypol", data=gain_ypol)

    print(f"Wrote gains to {output_gaintable_path}")


if __name__ == "__main__":
    main()
