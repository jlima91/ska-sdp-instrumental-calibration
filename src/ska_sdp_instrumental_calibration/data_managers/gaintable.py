from typing import Literal, Union

import dask.array as da
import numpy as np
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility

from .solution_interval import SolutionIntervals


def apply_antenna_gains_to_visibility(
    vis: np.ndarray,
    gains: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    inverse=False,
) -> np.ndarray:
    """
    vis: (time, baselineid, frequency, polarisation)
    gains: (time, antennas, frequency, nrec1, nrec2)
    antenna1: (baselineid)
        Indices of the antenna1 in all baseline pairs
    antenna2: (baselineid)
        Indices of the antenna2 in all baseline pairs
    inverse: bool
        Whether to inverse the gains before applying

    Returns
    -------
    np.ndarray of shape (time, baselineid, frequency, polarisation)
    """
    if inverse:
        gains = np.linalg.pinv(gains)

    vis_old_shape = vis.shape
    vis_new_shape = vis.shape[:3] + (2, 2)

    return np.einsum(  # pylint: disable=too-many-function-args
        "tbfpx,tbfxy,tbfqy->tbfpq",
        gains[:, antenna1, ...],
        vis.reshape(vis_new_shape),
        gains[:, antenna2, ...].conj(),
    ).reshape(vis_old_shape)


def create_gaintable_from_visibility(
    vis: Visibility,
    timeslice: Union[float, Literal["auto", "full"], None] = None,
    jones_type: Literal["T", "G", "B"] = "T",
    lower_precision: bool = True,
):
    """
    Similar behavior as create_gaintable_from_vis, except
    1. new param: timeslice="full"
        Will create a single solution across the entire observation time
    2. new param: lower_precision
        Ability to toggle precision of the data variables, currently
        between 4 or 8 bytes
    """
    # Backward compatibility. Should be removed as "auto" is vary vague
    if timeslice == "auto":
        timeslice = None

    soln_intervals = SolutionIntervals(vis.time.data, timeslice)
    ntimes = soln_intervals.size

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

    gain = da.broadcast_to(da.eye(nrec, dtype=complex_dtype), gain_shape)
    gain_weight = da.ones(gain_shape, dtype=float_dtype)
    gain_residual = da.zeros(
        [ntimes, nfrequency, nrec, nrec], dtype=float_dtype
    )

    gain_table = GainTable.constructor(
        gain=gain,
        time=soln_intervals.solution_time,
        interval=soln_intervals.intervals,
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
    gain_table.attrs["soln_interval_slices"] = soln_intervals.indices

    return gain_table


def reset_gaintable(gaintable: GainTable) -> GainTable:
    """
    Returns a new dask-backed gaintable with all data variables resetted
    to their initial sensible values
    """
    gain_shape = gaintable.gain.shape
    nrec = gain_shape[-1]
    gain = da.broadcast_to(
        da.eye(nrec, dtype=gaintable.gain.dtype), gain_shape
    )

    weight = da.ones(gaintable.weight.shape, dtype=gaintable.weight.dtype)

    residual = da.zeros(
        gaintable.residual.shape, dtype=gaintable.residual.dtype
    )

    # Deepcopy and change data variables
    # Simpler and less prone to errors than "assign"
    new_gaintable = gaintable.copy(deep=True)
    new_gaintable.gain.data = gain
    new_gaintable.weight.data = weight
    new_gaintable.residual.data = residual

    return new_gaintable
