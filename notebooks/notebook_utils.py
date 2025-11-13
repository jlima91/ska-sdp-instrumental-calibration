from enum import Enum
from typing import Literal, Union

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.workflow.utils import (
    create_grouped_array,
    get_indices_from_grouped_bins,
    get_intervals_from_grouped_bins,
)

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


class Backend(Enum):
    NUMPY = "numpy"
    DASK = "dask"

    @property
    def xp(self):
        if self is Backend.NUMPY:
            return np
        else:
            return da


def create_gaintable_from_vis_new(
    vis: Visibility,
    timeslice: Union[float, Literal["auto", "full"], None] = None,
    jones_type: Literal["T", "G", "B"] = "T",
    lower_precision: bool = True,
    backend: Backend = Backend.DASK,
):
    """
    Similar behavior as create_gaintable_from_vis, except
    1. new param: timeslice="full"
        Will create a single solution across the entire observation time
    2. new param: lower_precision
        Ability to toggle precision of the data variables, currently between 4 or 8 bytes
    3. new param: backend
        Option to choose between backends for data variables
    """
    # Backward compatibility. Should be removed as "auto" is vary vague
    if timeslice == "auto":
        timeslice = None

    # TODO: review this time slice creation logic
    gain_time_bins = create_grouped_array(vis.time, "time", timeslice)
    gain_time = gain_time_bins.mean().data
    gain_interval = get_intervals_from_grouped_bins(gain_time_bins)
    ntimes = len(gain_time)

    nants = vis.visibility_acc.nants

    # Set the frequency sampling
    if jones_type == "B":
        gain_frequency = vis.frequency.data
        nfrequency = len(gain_frequency)
    elif jones_type in ("G", "T"):
        gain_frequency = np.mean(vis.frequency.data, keepdims=True)
        nfrequency = 1
    else:
        raise ValueError(f"Unknown Jones type {jones_type}")

    # There is only one receptor frame in Visibility
    # Use it for both receptor1 and receptor2
    receptor_frame = ReceptorFrame(vis.visibility_acc.polarisation_frame.type)
    nrec = receptor_frame.nrec

    gain_shape = [ntimes, nants, nfrequency, nrec, nrec]

    # Create data variables with provided precision and backend
    if lower_precision:
        complex_dtype, float_dtype = np.complex64, np.float32
    else:
        complex_dtype, float_dtype = np.complex128, np.float64

    xp = backend.xp
    gain = xp.broadcast_to(xp.eye(nrec, dtype=complex_dtype), gain_shape)
    gain_weight = xp.ones(gain_shape, dtype=float_dtype)
    gain_residual = xp.zeros([ntimes, nfrequency, nrec, nrec], dtype=float_dtype)

    gain_table = GainTable.constructor(
        gain=gain,
        time=gain_time,
        interval=gain_interval,
        weight=gain_weight,
        residual=gain_residual,
        frequency=gain_frequency,
        receptor_frame=receptor_frame,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        jones_type=jones_type,
    )

    # Chunk data variables
    gain_table = gain_table.chunk(time=1)
    if gain_table.frequency.size == vis.frequency.size:
        gain_table = gain_table.chunk(frequency=vis.chunksizes["frequency"])

    # Attach solution interval slices as attribute
    gain_table.attrs["soln_interval_slices"] = get_indices_from_grouped_bins(
        gain_time_bins
    )

    return gain_table


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
