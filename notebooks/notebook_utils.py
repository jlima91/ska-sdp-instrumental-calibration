from typing import Literal

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger(__name__)

# Expected (simulated) gaintable has one extra channel at the end
# Need to remove it before comparing
REMOVE_LAST_ITEM = slice(None, -1, None)


def get_uv_wave(uvw, frequency):
    c = 3e8
    wavelength = c / frequency
    uvw_t = uvw.transpose("spatial", "time", "baselineid")
    return ((uvw_t[0] ** 2 + uvw_t[1] ** 2) ** 0.5) / wavelength


def plot_amp_uv_wave(input_vis, model_vis, prefix_path):
    fig = plt.figure(layout="constrained", figsize=(10, 5))
    fig.suptitle("Amp vs UVWave", fontsize=16)
    input_fig, model_fig = fig.subplots(1, 2)

    input_fig.set_ylim(0, 100)
    input_fig.set_title("Input visibilities")
    input_fig.set_xlabel("UVwave (λ)")
    input_fig.set_ylabel("amp")
    input_fig.scatter(
        abs(
            get_uv_wave(input_vis.uvw, input_vis.frequency).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        abs(
            input_vis.vis.isel(polarisation=0).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        s=1.0,
    )

    model_fig.set_ylim(0, 100)
    model_fig.set_title("Inst Predicted Model visibilitites")
    model_fig.set_xlabel("UVwave (λ)")
    model_fig.set_ylabel("amp")
    model_fig.scatter(
        abs(
            get_uv_wave(model_vis.uvw, model_vis.frequency).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        abs(
            model_vis.vis.isel(polarisation=0).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        s=1.0,
    )

    fig.savefig(f"{prefix_path}/amp-uvwave.png")
    plt.close(fig)


def plot_amp_freq(
    input_vis, model_vis, time_step, start_baseline, end_baseline, prefix_path
):
    fig = plt.figure(layout="constrained", figsize=(10, 5))
    fig.suptitle("Amp vs Frequency", fontsize=16)
    xx_ax, yy_ax = fig.subplots(1, 2)

    xx_ax.set_title("Model XX")
    xx_ax.set_xlabel("Channel")
    xx_ax.set_ylabel("Amp")

    yy_ax.set_title("Model YY")
    yy_ax.set_xlabel("Channel")
    yy_ax.set_ylabel("Amp")
    baselines = input_vis.baselineid.values

    for i in range(start_baseline, end_baseline):
        xx_ax.plot(
            abs(model_vis.vis.isel(time=time_step, baselineid=i, polarisation=0)),
            label=baselines[i],
        )
        yy_ax.plot(
            abs(model_vis.vis.isel(time=time_step, baselineid=i, polarisation=3)),
            label=baselines[i],
        )

    handles, labels = xx_ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Baselines", loc="outside center right")
    fig.savefig(f"{prefix_path}/amp-freq.png")

    plt.close(fig)


class H5ParmIO:
    @staticmethod
    def get_frequency(h5parm_path):
        with h5py.File(h5parm_path) as act_gain_f:
            frequency = act_gain_f["sol000"]["amplitude000"]["freq"][:]
            return frequency

    @staticmethod
    def get_polarisations(h5parm_path):
        with h5py.File(h5parm_path) as act_gain_f:
            pols = act_gain_f["sol000"]["amplitude000"]["pol"][:]
            pols = [item.decode("utf-8") for item in pols]
            return pols

    @staticmethod
    def get_antennas(h5parm_path):
        with h5py.File(h5parm_path) as act_gain_f:
            stations = act_gain_f["sol000"]["amplitude000"]["ant"][:]
            stations = [item.decode("utf-8") for item in stations]
            return stations

    @staticmethod
    def get_values(
        h5parm_path,
        solset="amplitude000",
        time: slice = slice(None),
        antenna: slice = slice(None),
        frequency: slice = slice(None),
        pol: slice = slice(None),
    ):
        with h5py.File(h5parm_path) as act_gain_f:
            vals = act_gain_f["sol000"][solset]["val"][time, antenna, frequency, pol][:]
            return vals


def check_for_nans(data: np.ndarray | da.Array):
    check = np.isnan(data).any()

    if isinstance(data, da.Array):
        return check.compute()

    return check


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol=1e-6,
    atol=1e-12,
    meta="Data",
    output: Literal["print", "log", "raise"] = "print",
):
    """
    Compare two arrays (NumPy or Dask) element-wise with tolerance thresholds.

    This function checks absolute and relative differences between two arrays and
    reports the maximum absolute difference, maximum relative difference, and the
    number and percentage of elements that differ beyond the specified tolerances.

    If either input is a Dask array, all required values are finalized in a single
    `dask.compute` call to minimize graph execution overhead.

    Parameters
    ----------
    actual : numpy.ndarray or dask.array.Array
        The array containing computed or observed values.
    expected : numpy.ndarray or dask.array.Array
        The reference array to compare against.
    rtol : float, default 1e-6
        Relative tolerance.
    atol : float, default 1e-12
        Absolute tolerance.
    meta : str, default "Data"
        Label used for output messages.
    output: str, Literal["print", "log", "raise"]
        Determines how to present the output
        print: Prints the comparision message
        log: Logs the comparision message
        raise: Raises AssertionError if values don't match
    """
    actions = {
        "print": lambda level, m: print(m),
        "log": lambda level, m: (
            logger.error(m) if level == "error" else logger.info(m)
        ),
        "raise": lambda level, m: (
            (_ for _ in ()).throw(AssertionError(m)) if level == "error" else None
        ),
    }

    is_dask = isinstance(actual, da.Array) or isinstance(expected, da.Array)

    if actual.dtype != expected.dtype:
        actions[output](
            "info",
            f"{meta}: dtype mismatch : actual: {actual.dtype}, expected: {expected.dtype}.\n"
            f"Comparison may be influenced by different precision.",
        )

    if check_for_nans(actual):
        actions[output](
            "info",
            f"{meta}: NaN found in actual. " f"Comparison may be incorrect.",
        )

    if check_for_nans(expected):
        actions[output](
            "info",
            f"{meta}: NaN found in expected. " f"Comparison may be incorrect.",
        )

    diff = actual - expected

    abs_diff = np.abs(diff)
    abs_diff_max = abs_diff.max()

    # element-wise relative difference (safe for expected == 0)
    rel_diff = abs_diff / np.maximum(np.abs(expected), 1e-30)
    rel_diff_max = rel_diff.max()

    # Decide rule for comparision
    # diff_mask = (abs_diff > atol) & (rel_diff > rtol) # More permissive
    diff_mask = abs_diff > (atol + rtol * np.abs(expected))  # numpy style comparision
    num_diff = diff_mask.sum()

    if is_dask:
        abs_diff_max, rel_diff_max, num_diff = da.compute(
            abs_diff_max, rel_diff_max, num_diff
        )

    total = actual.size

    if num_diff > 0:
        msg = (
            f"{meta}: do not match for atol={atol}, rtol={rtol}\n"
            f"\tmax abs diff = {abs_diff_max}\n"
            f"\tmax rel diff = {rel_diff_max}\n"
            f"\tdifferent elements: {num_diff} / {total} ({num_diff / total * 100:.6f}%)"
        )
        actions[output]("error", msg)
    else:
        msg = f"{meta}: match within atol={atol}, rtol={rtol}"
        actions[output]("info", msg)


# # TODO: Move this to its own test module
# # Test for the new gain interval logic
# timeslice = "full"
# gain_time_bins = create_solint_slices(vis.time, timeslice)
# gain_time = gain_time_bins.mean().data
# gain_interval = get_intervals_from_grouped_bins(gain_time_bins)
# idx = 0
# time = gain_time[idx]
# time_slice = {
#         "time": slice(
#             time - gain_interval[idx] / 2,
#             time + gain_interval[idx] / 2,
#         )
#     }
# assert np.all(vis.time.sel(time_slice).data == vis.time.data)


def identify_max_min_baselineid(uvw):
    """
    Identify the baseline IDs with the maximum and minimum baseline
    lengths.
    """
    bl_len = np.sqrt(
        uvw.sel(spatial="u") ** 2
        + uvw.sel(spatial="v") ** 2
        + uvw.sel(spatial="w") ** 2
    )

    bl_len_max = bl_len.max(dim="time")
    bl_len_min = bl_len.min(dim="time")

    valid = bl_len_max > 1.0
    bl_len_max = bl_len_max.where(valid)
    bl_len_min = bl_len_min.where(valid)

    return (
        bl_len_max.argmax(dim="baselineid").values,
        bl_len_min.argmin(dim="baselineid").values,
    )


def plot_phase_vs_time(input_vis, corrected_vis, channel, baseline, prefix_path):
    """
    Plot phase vs time for a given channel and baseline.
    """
    fig = plt.figure(layout="constrained", figsize=(10, 5))
    fig.suptitle("Phase vs Time", fontsize=16)
    xx_ax, yy_ax = fig.subplots(1, 2)

    xx_ax.set_title("Input")
    xx_ax.set_xlabel("Time (sec)")
    xx_ax.set_ylabel("Phase (deg)")
    xx_ax.set_ylim([-180, 180])

    yy_ax.set_title("Corrected")
    yy_ax.set_xlabel("Time (sec)")
    yy_ax.set_ylabel("Phase (deg)")
    yy_ax.set_ylim([-180, 180])

    xx_ax.scatter(
        input_vis.time,
        np.angle(
            input_vis.vis.isel(frequency=channel, baselineid=baseline, polarisation=0),
            deg=True,
        ),
    )
    yy_ax.scatter(
        corrected_vis.time,
        np.angle(
            corrected_vis.vis.isel(
                frequency=channel, baselineid=baseline, polarisation=0
            ),
            deg=True,
        ),
    )

    fig.savefig(f"{prefix_path}/phase-time-{channel}-{baseline}.png")

    plt.close(fig)


def plot_time_vs_freq_for_phase(input_vis, corrected_vis, baseline, prefix_path):
    """
    Plot waterfallplot for time vs frequency for phase for original and corrected visibilities for a given baseline.

    Parameters
    ----------
    input_vis : xarray.Dataset
        The input visibility dataset.
    corrected_vis : xarray.Dataset
        The corrected visibility dataset.
    baseline : int
        The baseline ID to plot.
    prefix_path : str
        The path prefix to save the plot.
    """
    _input_vis = input_vis.vis.isel(baselineid=baseline, polarisation=0)
    input_x = _input_vis.time
    input_y = _input_vis.frequency

    input_X, input_Y = np.meshgrid(input_x, input_y)
    input_Z = np.angle(_input_vis.T)

    _corrected_vis = corrected_vis.vis.isel(baselineid=baseline, polarisation=0)
    corrected_x = _corrected_vis.time
    corrected_y = _corrected_vis.frequency

    corrected_X, corrected_Y = np.meshgrid(corrected_x, corrected_y)
    corrected_Z = np.angle(_corrected_vis.T, deg=True)

    fig = plt.figure(layout="constrained", figsize=(10, 5))
    fig.suptitle("Time Vs Freq for Phase", fontsize=16)

    input_plt, corrected_plt = fig.subplots(1, 2)

    input_plt.set_title("Input")
    input_plt.set_xlabel("Time (sec)")
    input_plt.set_ylabel("Freq (Hz)")
    pcm = input_plt.pcolormesh(input_X, input_Y, input_Z)

    corrected_plt.set_title("Corrected")
    corrected_plt.set_xlabel("Time (sec)")
    corrected_plt.set_ylabel("Freq (Hz)")
    pcm = corrected_plt.pcolormesh(corrected_X, corrected_Y, corrected_Z)

    plt.colorbar(pcm, label="Phase (deg)")

    fig.savefig(f"{prefix_path}/phase-time-freq-phase-waterfall-{baseline}.png")

    plt.close(fig)


def plot_time_vs_freq_for_phase_multiple_baselines(
    vis, baseline_start, baseline_end, title_suffix, prefix_path
):
    """
    Plot waterfallplot for time vs frequency for phase for multiple baselines.

    Parameters
    ----------
    vis : xarray.Dataset
        The visibility dataset.
    baseline_start : int
        The starting baseline ID to plot.
    baseline_end : int
        The ending baseline ID to plot.
    title_suffix : str
        Suffix to add to the plot title and filename.
    prefix_path : str
        The path prefix to save the plot.
    """
    _vis = vis.vis.isel(polarisation=0)

    time = _vis.time
    freq = _vis.frequency
    time_X, freq_Y = np.meshgrid(time, freq)

    baselines = range(baseline_start, baseline_end + 1)

    phases_list = [
        np.angle(_vis.isel(baselineid=baseline).T, deg=True) for baseline in baselines
    ]
    baseline_counts = len(baselines)
    cols = 5
    rows = baseline_counts // cols + baseline_counts % cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(18, 12), constrained_layout=True, sharex=True, sharey=True
    )
    fig.suptitle(f"Time Vs Freq for Phase {title_suffix}", fontsize=16)

    for ax, phase in zip(axes.flat, phases_list):
        pcm = ax.pcolormesh(time_X, freq_Y, phase, shading="auto")

    fig.supxlabel("Time (sec)")
    fig.supylabel("Freq (Hz)")

    fig.colorbar(pcm, ax=axes, label="Phase (deg)", shrink=0.85)

    fig.savefig(
        f"{prefix_path}/phase-time-freq-phase-waterfall-{baseline_start}-{baseline_end}-{title_suffix}.png"
    )

    plt.close(fig)
