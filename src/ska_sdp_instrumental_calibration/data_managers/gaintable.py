from typing import Literal, Union

import dask.array as da
import numpy as np
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility

from .solution_interval import SolutionIntervals


def create_gaintable_from_visibility(
    vis: Visibility,
    timeslice: Union[float, Literal["auto", "full"], None] = None,
    jones_type: Literal["T", "G", "B"] = "T",
    lower_precision: bool = True,
    skip_default_chunk: bool = False,
):
    """
    Create gaintable from visibility.
    Similar behavior as `create_gaintable_from_vis`, except
    1. new param: timeslice="full"
        Will create a single solution across the entire observation time
    2. new param: lower_precision
        Ability to toggle precision of the data variables, currently
        between 4 or 8 bytes

    Parameters
    ----------
    vis: Visibility
        Visibility to create gaintable from
    timeslice: str|float
        Time slice definition ot be used while creating the gaintable
        Default: None
    jones_type: str
        Jones types for the gaintable.
        Allowed valued: "T", "G", "B"
        Default: "T"
    lower_precision: bool
        Used to set up the float bit sizes while initialising the gaintable.
        If true, uses np.complex64 and np.float32 instead of higher precision
        np.complex128 and np.float64. Useful for memory optimization.
        Default: True
    skip_default_chunk: bool
        If set to true, skips Dask/Xarray chunking of data in alignment to the
        input visibility. Useful in cases of chunk alignment issues.
        Default: False

    Returns
    -------
        GainTable
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

    # Attach solution interval slices as attribute
    gain_table.attrs["soln_interval_slices"] = soln_intervals.indices
    # Chunk data variables

    if skip_default_chunk:
        return gain_table

    gain_table = gain_table.chunk(time=1)
    if gain_table.frequency.size == vis.frequency.size:
        gain_table = gain_table.chunk(frequency=vis.chunksizes["frequency"])

    return gain_table


def reset_gaintable(gaintable: GainTable) -> GainTable:
    """
    Returns a new dask-backed gaintable with all data variables resetted
    to their initial sensible values

    Parameters
    ----------
    gaintable: GainTable
        Gaintable object to be reset

    Returns
    -------
    Gaintable with data variables resetted to their intial sensible values.
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
