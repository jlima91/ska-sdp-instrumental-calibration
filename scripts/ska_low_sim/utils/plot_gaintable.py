#!/usr/bin/env python3
"""
Generate diagnostic plots of OSKAR gaintables stored in h5 file format.
Plots are saved as PNG files in a new subfolder.

Authored by:
- Team Dhruva

Usage:
    python plot_gains.py gain_model_scan_0.h5 ./output
"""

import argparse
import sys
import os
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt


def load_gains(filename):
    with h5py.File(filename, "r") as f:
        freqs = f["freq (Hz)"][:] / 1e6  # convert to MHz
        gain_x = f["gain_xpol"][:]  # shape [time, freq, station]
        gain_y = f["gain_ypol"][:]
    return freqs, gain_x, gain_y


def save_plot(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_amp_vs_freq(freqs, gains, outdir, time_idx=0, pol="X"):
    fig, ax = plt.subplots()
    nstations = gains.shape[2]
    for station_idx in range(0, nstations):
        amp = np.abs(gains[time_idx, :, station_idx])
        ax.plot(freqs, amp, label=f"Pol{pol}")
    ax.set_ylim(0, 2)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Amp vs Freq — All stations, time {time_idx}, pol{pol}")
    ax.grid()
    save_plot(fig, outdir, f"amp_vs_freq_pol{pol}_all_stations.png")


def plot_phase_vs_freq(freqs, gains, outdir, time_idx=0, pol="X"):
    fig, ax = plt.subplots()
    nstaions = gains.shape[2]
    for station_idx in range(nstaions):
        phase = np.unwrap(np.angle(gains[time_idx, :, station_idx])) * 180 / np.pi
        ax.plot(freqs, phase)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Phase (deg)")
    ax.set_title(f"Phase vs Freq — All stations, time {time_idx}, pol{pol}")
    ax.grid()
    save_plot(fig, outdir, f"phase_vs_freq_pol{pol}_all_station.png")


def plot_waterfall(freqs, gains, outdir, pol="X", sampling_time=1.0):
    time_axis = np.arange(gains.shape[0]) * sampling_time
    nstations = gains.shape[2]
    n_rows = 3
    n_cols = 3
    plots_per_group = n_rows * n_cols
    plot_groups = np.split(
        range(nstations),
        range(plots_per_group, nstations, plots_per_group),
    )

    for group_idx, station_group in enumerate(plot_groups):
        amp_fig, amp_axes = plt.subplots(
            n_rows, n_cols, figsize=(18, 18), constrained_layout=True
        )
        amp_axes = amp_axes.ravel()

        phase_fig, phase_axes = plt.subplots(
            n_rows, n_cols, figsize=(18, 18), constrained_layout=True
        )
        phase_axes = phase_axes.ravel()

        for sub_idx, station_idx in enumerate(station_group):
            ##Amp water plot per station
            amp_ax = amp_axes[sub_idx]
            amp_tf = np.abs(gains[:, :, station_idx])
            amp_im = amp_ax.imshow(
                amp_tf.T,
                aspect="auto",
                origin="lower",
                extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]],
                cmap="viridis",
            )
            amp_ax.set_title(f"Station {station_idx}")
            amp_ax.set_xlabel("Time (s)")
            amp_ax.set_ylabel("Frequency (MHz)")

            ##Phase water plot per station
            phase_ax = phase_axes[sub_idx]
            phase_tf = (
                np.unwrap(np.angle(gains[:, :, station_idx]), axis=0) * 180 / np.pi
            )
            phase_im = phase_ax.imshow(
                phase_tf.T,
                aspect="auto",
                origin="lower",
                extent=[time_axis[0], time_axis[-1], freqs[0], freqs[-1]],
                cmap="viridis",
            )
            phase_ax.set_title(f"Station {station_idx}")
            phase_ax.set_xlabel("Time (s)")
            phase_ax.set_ylabel("Frequency (MHz)")

        # hide unused axes (if nstations not multiple of 9)
        for ax_indx in range(len(station_group), n_rows * n_cols):
            amp_axes[ax_indx].axis("off")
            phase_axes[ax_indx].axis("off")

        amp_fig.colorbar(
            amp_im,
            ax=amp_axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.02,
            label="Amplitude",
        )
        amp_fig.suptitle(
            f"Amp Waterfall - Stations {station_group[0]} TO {station_group[-1]}, polX",
            fontsize=16,
        )
        save_plot(
            amp_fig,
            outdir,
            f"amp_waterfall_stations{station_group[0]}-{station_group[-1]}.png",
        )

        phase_fig.colorbar(
            phase_im,
            ax=phase_axes,
            orientation="vertical",
            fraction=0.02,
            pad=0.02,
            label="Phase",
        )
        phase_fig.suptitle(
            f"Phase Waterfall - Stations {station_group[0]} TO {station_group[-1]}, polX",
            fontsize=16,
        )
        save_plot(
            phase_fig,
            outdir,
            f"phase_waterfall_stations{station_group[0]}-{station_group[-1]}.png",
        )


if __name__ == "__main__":
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

    args = parser.parse_args()
    outdir = args.outdir or f"{os.path.basename(args.oskar_gaintable)}_plots"
    os.makedirs(outdir, exist_ok=True)

    freqs, gx1, gy1 = load_gains(args.oskar_gaintable)
    sampling_time = 1.0  # seconds; adjust if known

    plot_amp_vs_freq(freqs, gx1, outdir, pol="X")
    plot_phase_vs_freq(freqs, gx1, outdir, pol="X")
    plot_waterfall(freqs, gx1, outdir, pol="X", sampling_time=sampling_time)

    print(f"Plots saved in folder: {outdir}")
