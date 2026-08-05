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
- Gain outliers injected at explicit (station, channel) cells.
"""

import logging
import os
import random
import subprocess
import sys

import h5py
import numpy as np
import yaml
from resources import (  # pylint: disable=import-error
    H5PARM_CONVERTER_SCRIPT,
    SPLINE_DATA_PATH,
    TEL_MODEL,
)
from scipy.interpolate import BSpline

from .constants import (
    CHANNEL_WIDTH_HZ,
    END_FREQ_HZ,
    N_STATIONS,
    OBSERVING_TIME_MINS,
    OUTLIER_AMPLITUDE,
    OUTLIER_CHANNEL_INDICES,
    OUTLIER_PHASE_DEG,
    OUTLIER_STATION_INDICES,
    SAMPLING_TIME_SEC,
    START_FREQ_HZ,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


logger = logging.getLogger("GAINTABLE-SIM")


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


def bandpass_amplitude(simulation_frequency_table, spline_data):
    """
    Calculate phase values for bandpass

    Returns
    -------
    station_bandpass_amplitude_X, station_bandpass_amplitude_Y
        np.ndarray of shape (n_channels,)
    """
    knots_X = spline_data["knots_X"]
    coefficients_X = spline_data["coefficients_X"]
    degree_X = spline_data["degree_X"]
    knots_Y = spline_data["knots_Y"]
    coefficients_Y = spline_data["coefficients_Y"]
    degree_Y = spline_data["degree_Y"]

    spline_X = BSpline(knots_X, coefficients_X, degree_X)
    spline_Y = BSpline(knots_Y, coefficients_Y, degree_Y)
    station_bandpass_amplitude_X = spline_X(simulation_frequency_table)
    station_bandpass_amplitude_Y = spline_Y(simulation_frequency_table)

    return station_bandpass_amplitude_X, station_bandpass_amplitude_Y


def bandpass_phase(
    simulation_start_frequency,
    simulation_end_frequency,
    simulation_frequency_table,
    spline_data,
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
    start_idx = np.where(spline_frequencies == simulation_start_frequency)[0][
        0
    ]
    end_idx = (
        np.where(spline_frequencies == simulation_end_frequency)[0][0] + 1
    )

    station_bandpass_phase_X = np.interp(
        simulation_frequency_table,
        spline_frequencies[start_idx:end_idx],
        phase_X[start_idx:end_idx],
    )
    station_bandpass_phase_Y = np.interp(
        simulation_frequency_table,
        spline_frequencies[start_idx:end_idx],
        phase_Y[start_idx:end_idx],
    )

    return station_bandpass_phase_X, station_bandpass_phase_Y


def add_gain_outliers(
    gain, station_indices, channel_indices, amplitude, phase_deg
):
    """
    Add complex gain outliers by corrupting an explicit set of stations
    and channels.

    Parameters
    ----------
    gain : np.ndarray
        Complex gain array with shape (n_time, n_channels, n_stations)
    station_indices : sequence of int
        Stations to corrupt
    channel_indices : sequence of int
        Channels to corrupt, for every station in station_indices
    amplitude : float
        Amplitude multiplier applied to every corrupted cell
    phase_deg : float
        Phase offset (degrees) applied to every corrupted cell

    Returns
    -------
    gain_with_outliers : np.ndarray
        Complex gain array with outliers injected
    """
    gain = gain.copy()
    corruption = amplitude * np.exp(1j * np.deg2rad(phase_deg))

    for station in station_indices:
        for channel in channel_indices:
            gain[:, channel, station] *= corruption

    return gain


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


def time_variant_effects(
    calibration_time, n_stations, number_of_cal_time_samples
):
    """
    Calculates time variatnt effects added to the bandpass

    Returns
    -------
    phase_time_variation_profile
        np.ndarrays[float] of shape (time_samples, n_stations)
    """

    phase_time_variation_profile = np.zeros(
        (len(calibration_time), n_stations), dtype=np.float64
    )
    phase_variation_frequency = 1 / create_gaussian_noise(
        n_stations, number_of_cal_time_samples * 0.5, 30
    )  # 0.5 is the number of scans in 3000 seconds with 30 samples sigma.

    phase_function_amplitude = 1  # Amplitude of function.
    phase_function_phase = create_gaussian_noise(
        n_stations, 0, 0.2
    )  # Phase of function in radians.
    phase_fuction_offset = 0  # Offset of function.

    phase_time_variation_profile = phase_fuction_offset + (
        phase_function_amplitude
        * np.sin(
            2
            * np.pi
            * phase_variation_frequency
            * calibration_time[:, np.newaxis]
            + phase_function_phase
        )
    )

    return phase_time_variation_profile


def calculate_time_variant_phase(
    n_stations,
    start_frequency_hz,
    end_frequency_hz,
    channel_bandwidth_hz,
    sampling_time_sec,
    observing_time_mins,
):
    # -------------- Unpack parameters ---------------- #
    simulation_start_frequency = start_frequency_hz * 1e-6
    simulation_end_frequency = end_frequency_hz * 1e-6
    correlated_channel_bandwidth = channel_bandwidth_hz * 1e-6
    observing_time_cal = observing_time_mins * 60

    # ---------------- Setup ---------------------#

    number_of_cal_time_samples = int(observing_time_cal // sampling_time_sec)

    # TODO: Verify what is the correct way to call arange
    simulation_frequency_table = np.arange(
        simulation_start_frequency,
        simulation_end_frequency,
        correlated_channel_bandwidth,
    )

    shape = (
        number_of_cal_time_samples,
        len(simulation_frequency_table),
        n_stations,
    )

    calibration_time = np.arange(
        0, number_of_cal_time_samples * sampling_time_sec, sampling_time_sec
    )

    phase_time_variation_profile = time_variant_effects(
        calibration_time=calibration_time,
        n_stations=n_stations,
        number_of_cal_time_samples=number_of_cal_time_samples,
    )

    gain_xpol = np.broadcast_to(
        (np.exp(1j * phase_time_variation_profile))[:, np.newaxis, :],
        shape,
    ).copy()

    gain_ypol = np.broadcast_to(
        (np.exp(1j * phase_time_variation_profile))[:, np.newaxis, :],
        shape,
    ).copy()

    return gain_xpol, gain_ypol, simulation_frequency_table


def calculate_gains(
    n_stations,
    start_frequency_hz,
    end_frequency_hz,
    channel_bandwidth_hz,
    sampling_time_sec,
    observing_time_mins,
    outlier_station_indices,
    outlier_channel_indices,
    outlier_amplitude,
    outlier_phase_deg,
):
    # -------------- Unpack parameters ---------------- #
    simulation_start_frequency = start_frequency_hz * 1e-6
    simulation_end_frequency = end_frequency_hz * 1e-6
    correlated_channel_bandwidth = channel_bandwidth_hz * 1e-6
    observing_time_cal = observing_time_mins * 60

    # ---------------- Setup ---------------------#
    spline_data = np.load(SPLINE_DATA_PATH)

    number_of_cal_time_samples = int(observing_time_cal // sampling_time_sec)

    # TODO: Verify what is the correct way to call arange
    simulation_frequency_table = np.arange(
        simulation_start_frequency,
        simulation_end_frequency,
        correlated_channel_bandwidth,
    )

    # ----------------- Start calculating gains ------------ #

    station_bandpass_amplitude_X, station_bandpass_amplitude_Y = (
        bandpass_amplitude(
            simulation_frequency_table=simulation_frequency_table,
            spline_data=spline_data,
        )
    )
    station_bandpass_phase_X, station_bandpass_phase_Y = bandpass_phase(
        simulation_start_frequency=simulation_start_frequency,
        simulation_end_frequency=simulation_end_frequency,
        simulation_frequency_table=simulation_frequency_table,
        spline_data=spline_data,
    )

    shape = (
        number_of_cal_time_samples,
        len(simulation_frequency_table),
        n_stations,
    )
    gain_xpol = np.broadcast_to(
        (
            station_bandpass_amplitude_X
            * np.exp(1j * np.deg2rad(station_bandpass_phase_X))
        )[np.newaxis, :, np.newaxis],
        shape,
    ).copy()
    gain_ypol = np.broadcast_to(
        (
            station_bandpass_amplitude_Y
            * np.exp(1j * np.deg2rad(station_bandpass_phase_Y))
        )[np.newaxis, :, np.newaxis],
        shape,
    ).copy()

    gain_xpol = add_gain_outliers(
        gain_xpol,
        outlier_station_indices,
        outlier_channel_indices,
        outlier_amplitude,
        outlier_phase_deg,
    )
    gain_ypol = add_gain_outliers(
        gain_ypol,
        outlier_station_indices,
        outlier_channel_indices,
        outlier_amplitude,
        outlier_phase_deg,
    )

    return gain_xpol, gain_ypol, simulation_frequency_table


def convert_gaintable_to_h5parm(output_dir, gaintable_path, start_time):
    """Convert the OSKAR-format gaintable h5 into a DP3-compatible H5Parm
    using the ska_low_sim converter script.
    """
    converter_config = {
        "tel_model": str(TEL_MODEL),
        "sampling_time_sec": SAMPLING_TIME_SEC,
        "fields": {"EoR2": {"Cal1": {"transit_time": str(start_time)}}},
    }
    config_path = os.path.join(output_dir, "h5parm_converter_config.yaml")
    with open(config_path, mode="w", encoding="utf-8") as config_file:
        yaml.safe_dump(converter_config, config_file)

    h5parm_path = os.path.join(output_dir, "sim_gaintable.h5parm")
    command = [
        sys.executable,
        str(H5PARM_CONVERTER_SCRIPT),
        config_path,
        str(gaintable_path),
        h5parm_path,
    ]
    logger.info("Converting gaintable to H5Parm: %s", command)
    subprocess.run(command, check=True)

    return h5parm_path


def generate_bandpass_gaintable(output_dir, start_time):
    """Generate a synthetic bandpass gaintable and convert it into a
    DP3-compatible H5Parm.

    Returns the path of the H5Parm file.
    """
    gain_xpol, gain_ypol, sim_freqs = calculate_gains(
        n_stations=N_STATIONS,
        start_frequency_hz=START_FREQ_HZ,
        end_frequency_hz=END_FREQ_HZ,
        channel_bandwidth_hz=CHANNEL_WIDTH_HZ,
        sampling_time_sec=SAMPLING_TIME_SEC,
        observing_time_mins=OBSERVING_TIME_MINS,
        outlier_station_indices=OUTLIER_STATION_INDICES,
        outlier_channel_indices=OUTLIER_CHANNEL_INDICES,
        outlier_amplitude=OUTLIER_AMPLITUDE,
        outlier_phase_deg=OUTLIER_PHASE_DEG,
    )

    gaintable_path = os.path.join(output_dir, "sim_gaintable.h5")
    with h5py.File(gaintable_path, "w") as f:
        f.create_dataset("freq (Hz)", data=sim_freqs * 1e6)
        f.create_dataset("gain_xpol", data=gain_xpol)
        f.create_dataset("gain_ypol", data=gain_ypol)

    logger.info("Finished gaintable generation, path: %s", gaintable_path)

    return convert_gaintable_to_h5parm(output_dir, gaintable_path, start_time)


def generate_target_gaintable(output_dir, start_time):
    """Generate a synthetic bandpass gaintable and convert it into a
    DP3-compatible H5Parm.

    Returns the path of the H5Parm file.
    """

    gain_xpol, gain_ypol, sim_freqs = calculate_time_variant_phase(
        n_stations=N_STATIONS,
        start_frequency_hz=START_FREQ_HZ,
        end_frequency_hz=END_FREQ_HZ,
        channel_bandwidth_hz=CHANNEL_WIDTH_HZ,
        sampling_time_sec=SAMPLING_TIME_SEC,
        observing_time_mins=OBSERVING_TIME_MINS,
    )

    gaintable_path = os.path.join(output_dir, "sim_gaintable.h5")
    with h5py.File(gaintable_path, "w") as f:
        f.create_dataset("freq (Hz)", data=sim_freqs * 1e6)
        f.create_dataset("gain_xpol", data=gain_xpol)
        f.create_dataset("gain_ypol", data=gain_ypol)

    logger.info("Finished gaintable generation, path: %s", gaintable_path)

    return convert_gaintable_to_h5parm(output_dir, gaintable_path, start_time)
