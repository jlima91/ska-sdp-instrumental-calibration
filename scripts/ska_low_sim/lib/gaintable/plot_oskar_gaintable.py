#!/usr/bin/env python3
"""
Generate diagnostic plots of OSKAR gaintables stored in h5 file format.
Plots are saved as PNG files in a new subfolder.

Authored by:
- Team Dhruva

Usage:
    python plot_oskar_gaintable.py gain_model_scan_0.h5 ./output
"""

import argparse
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt

# Stations per waterfall figure
N_ROWS = 3
N_COLS = 3


def load_gains(filename):
    with h5py.File(filename, "r") as f:
        freqs = f["freq (Hz)"][:] / 1e6  # convert to MHz
        gain_x = f["gain_xpol"][:]  # shape [time, freq, station]
        gain_y = f["gain_ypol"][:]
    return freqs, gain_x, gain_y


def save_plot(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def phase_deg(gains, axis):
    return np.unwrap(np.angle(gains), axis=axis) * 180 / np.pi


def plot_amp_vs_freq(freqs, gains, outdir, time_idx=0, pol="X"):
    fig, ax = plt.subplots()
    for station_idx in range(gains.shape[2]):
        ax.plot(freqs, np.abs(gains[time_idx, :, station_idx]))
    ax.set_ylim(0, 2)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Amp vs Freq — All stations, time {time_idx}, pol{pol}")
    ax.grid()
    save_plot(fig, outdir, f"amp_vs_freq_pol{pol}_all_stations.png")


def plot_phase_vs_freq(freqs, gains, outdir, time_idx=0, pol="X"):
    fig, ax = plt.subplots()
    for station_idx in range(gains.shape[2]):
        ax.plot(freqs, phase_deg(gains[time_idx, :, station_idx], axis=-1))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title(f"Phase vs Freq — All stations, time {time_idx}, pol{pol}")
    ax.grid()
    save_plot(fig, outdir, f"phase_vs_freq_pol{pol}_all_station.png")


def _waterfall_figure(
    values, station_group, extent, outdir, name, label, pol, file_prefix
):
    """
    One figure with a (time, frequency) waterfall per station.

    :param values: Real array with shape (time, freq, station).
    """
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(18, 18), constrained_layout=True)
    axes = axes.ravel()
    for sub_idx, station_idx in enumerate(station_group):
        ax = axes[sub_idx]
        im = ax.imshow(
            values[:, :, station_idx].T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
        )
        ax.set_title(f"Station {station_idx}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (MHz)")

    # hide unused axes (if nstations not multiple of 9)
    for ax in axes[len(station_group) :]:
        ax.axis("off")

    fig.colorbar(
        im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label=label
    )
    first, last = station_group[0], station_group[-1]
    fig.suptitle(
        f"{name} Waterfall - Stations {first} TO {last}, pol{pol}", fontsize=16
    )
    save_plot(fig, outdir, f"{file_prefix}_waterfall_stations{first}-{last}.png")


def plot_waterfall(freqs, gains, outdir, pol="X", sampling_time=1.0):
    time_axis = np.arange(gains.shape[0]) * sampling_time
    extent = [time_axis[0], time_axis[-1], freqs[0], freqs[-1]]
    nstations = gains.shape[2]
    plots_per_group = N_ROWS * N_COLS
    plot_groups = np.split(
        range(nstations),
        range(plots_per_group, nstations, plots_per_group),
    )
    amp = np.abs(gains)
    phase = phase_deg(gains, axis=0)
    for station_group in plot_groups:
        _waterfall_figure(
            amp, station_group, extent, outdir, "Amp", "Amplitude", pol, "amp"
        )
        _waterfall_figure(
            phase, station_group, extent, outdir, "Phase", "Phase", pol, "phase"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "oskar_gaintable", type=str, help="Path to OSKAR supported H5 file"
    )
    parser.add_argument(
        "outdir",
        nargs="?",
        type=str,
        default=None,
        help="Output directory for plots (default: <input_basename>_plots)",
    )
    parser.add_argument(
        "--sampling-time",
        type=float,
        default=1.0,
        help="Time between gain solutions (s), for the waterfall time axis",
    )
    args = parser.parse_args()

    outdir = args.outdir or f"{os.path.basename(args.oskar_gaintable)}_plots"
    os.makedirs(outdir, exist_ok=True)

    freqs, gain_x, _ = load_gains(args.oskar_gaintable)

    plot_amp_vs_freq(freqs, gain_x, outdir, pol="X")
    plot_phase_vs_freq(freqs, gain_x, outdir, pol="X")
    plot_waterfall(freqs, gain_x, outdir, pol="X", sampling_time=args.sampling_time)

    print(f"Plots saved in folder: {outdir}")


if __name__ == "__main__":
    main()
