import logging

import numpy as np
import xarray as xr
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.visibility import Visibility

from ..numpy_processors.gaintable import apply_antenna_gains_to_visibility

logger = logging.getLogger(__name__)


def _apply_gaintable_to_dataset_ufunc(
    vis: np.ndarray,
    gains: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    inverse: np.ndarray = False,
):
    """
    A bridge function between apply_gaintable_to_dataset and
    apply_antenna_gains_to_visibility

    Parameters
    ----------
    vis: (time, frequency, baselineid, polarisation)
    gains: (antennas, nrec1, nrec2) or (frequency, antennas, nrec1, nrec2)
    antenna1: (baselineid)
        Indices of the antenna1 in all baseline pairs
    antenna2: (baselineid)
        Indices of the antenna2 in all baseline pairs
    inverse: bool
        Whether to inverse the gains before applying

    Returns
    -------
    np.ndarray of shape (time, frequency, baselineid, polarisation)
    """
    # Add frequency dimension at the start in case its dropped before
    if len(gains.shape) == 3:
        gains = gains[np.newaxis, ...]

    # Add time axis as it is dropped in apply_ufunc
    gains = gains[np.newaxis, ...]

    return apply_antenna_gains_to_visibility(
        vis.transpose(0, 2, 1, 3),
        gains.transpose(0, 2, 1, 3, 4),
        antenna1,
        antenna2,
        inverse,
    ).transpose(0, 2, 1, 3)


def apply_gaintable_to_dataset(
    vis: Visibility,
    gaintable: GainTable,
    inverse=False,
) -> Visibility:
    """
    Apply calibration gains from a GainTable to a Visibility dataset.

    This function modifies visibility data by applying antenna-based Jones
    matrices. It supports both calibration (removing instrumental effects by
    applying inverse gains) and corruption (simulating instrumental effects
    by applying forward gains).

    The function handles different Jones matrix types:

    - Type 'B' (Bandpass): Gains match the visibility frequency resolution.
    - Type 'T' or 'G': Frequency-independent gains are broadcast across all
      visibility channels.

    Operations are performed per solution interval. The function iterates over
    time slices defined in the gaintable, applying the corresponding gain
    solution to the matching block of visibilities using
    :py:func:`xarray.apply_ufunc` for parallel processing.

    Parameters
    ----------
    vis : Visibility
        The input visibility dataset containing observed data.
    gaintable : GainTable
        The calibration solutions (Jones matrices) to apply. Must contain
        valid `soln_interval_slices` mapping gain times to visibility times.
    inverse : bool, optional
        If True, apply the inverse of the gains (calibrate).
        If False, apply the gains directly (corrupt/simulate).
        Default is False.

    Returns
    -------
    Visibility
        A new Visibility object containing the modified visibility data
        ('vis' variable updated) with attributes preserved.

    Raises
    ------
    AssertionError
        If Jones type is not 'B' and the gaintable has more than one
        frequency channel.
    """
    gains = gaintable.gain
    if gaintable.jones_type == "B":
        # solution frequency same as vis frequency
        # chunking just to be sure that they match
        gains = gains.chunk({"frequency": vis.chunksizes["frequency"]})
    else:  # jones_type == T or G
        assert gains.frequency.size == 1, "Gaintable frequency "
        "must either match to visibility frequency, or must be of size 1"
        # Remove frequency dimension for apply_ufunc to work properly
        gains = gains.isel(frequency=0, drop=True)

    soln_interval_slices = gaintable.soln_interval_slices

    applied_vis_across_solutions = []
    for idx, slc in enumerate(soln_interval_slices):
        applied_vis_per_soln_interval = xr.apply_ufunc(
            _apply_gaintable_to_dataset_ufunc,
            vis.vis.isel(time=slc),
            gains.isel(time=idx, drop=True),
            input_core_dims=[
                ["baselineid", "polarisation"],
                ["antenna", "receptor1", "receptor2"],
            ],
            output_core_dims=[
                ["baselineid", "polarisation"],
            ],
            dask="parallelized",
            output_dtypes=[vis.vis.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes={
                    "baselineid": vis.baselineid.size,
                    "polarisation": vis.polarisation.size,
                }
            ),
            kwargs={
                "antenna1": vis.antenna1,
                "antenna2": vis.antenna2,
                "inverse": inverse,
            },
        )
        applied_vis_per_soln_interval = (
            applied_vis_per_soln_interval.transpose(
                "time", "baselineid", "frequency", "polarisation"
            )
        )
        applied_vis_across_solutions.append(applied_vis_per_soln_interval)

    applied: xr.DataArray = xr.concat(applied_vis_across_solutions, dim="time")

    applied = applied.assign_attrs(vis.vis.attrs)
    return vis.assign({"vis": applied})
