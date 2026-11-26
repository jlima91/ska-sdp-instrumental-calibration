import logging

import numpy as np
import xarray as xr
from ska_sdp_func_python.calibration import apply_antenna_gains_to_visibility

from ..data_managers.processing_set import (
    get_baseline_antenna_indices,
    get_correlated_data,
    with_correlated_data,
)
from ..data_managers.schema.calibration_set import GainCalibrationSetXds
from ..data_managers.schema.polarization import pol_to_jones

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
    vis: (time, frequency, baseline_id, polarization)
    gains: (antennas, npol) or (frequency, antennas, npol)
    antenna1: (baseline_id)
        Indices of the antenna1 in all baseline pairs
    antenna2: (baseline_id)
        Indices of the antenna2 in all baseline pairs
    inverse: bool
        Whether to inverse the gains before applying

    Returns
    -------
    np.ndarray of shape (time, frequency, baseline_id, polarization)
    """
    gains = pol_to_jones(gains)

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
    ms_xdt: xr.DataTree,
    gaintable: GainCalibrationSetXds,
    inverse=False,
) -> xr.DataTree:
    """
    Apply calibration gains from a GainCalibrationSetXds to the correlated
    data of an MSv4 node.

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
    ms_xdt : xr.DataTree
        MSv4 node holding a single data group.
    gaintable : GainCalibrationSetXds
        The calibration solutions (Jones matrices) to apply. Must contain
        valid `soln_interval_slices` mapping gain times to visibility times,
        and have the antennas of ``ms_xdt`` (in ``antenna_xds`` order) and
        its polarizations.
    inverse : bool, optional
        If True, apply the inverse of the gains (calibrate).
        If False, apply the gains directly (corrupt/simulate).
        Default is False.

    Returns
    -------
    xr.DataTree
        Copy of ``ms_xdt`` with the gains applied to its correlated data.
        Other variables and child datasets are those of ``ms_xdt``.

    Raises
    ------
    AssertionError
        If Jones type is not 'B' and the gaintable has more than one
        frequency channel.
    ValueError
        If the gaintable antennas or polarizations differ from those of
        ``ms_xdt``.
    """
    correlated_data = get_correlated_data(ms_xdt)

    # Baseline antenna indices refer to the antenna_xds order, so the gains
    # must be in the same order.
    if not np.array_equal(
        gaintable.antenna_name.values,
        ms_xdt["antenna_xds"].ds.antenna_name.values,
    ):
        raise ValueError(
            "Gaintable antennas differ from the MSv4 antenna_xds antennas"
        )
    if not np.array_equal(
        gaintable.polarization.values, correlated_data.polarization.values
    ):
        raise ValueError(
            f"Gaintable polarizations {gaintable.polarization.values} "
            "differ from visibility polarizations "
            f"{correlated_data.polarization.values}"
        )

    gains = gaintable.CALPARAM_GAIN
    if gaintable.jones_type == "B":
        # solution frequency same as vis frequency
        # chunking just to be sure that they match.
        # Chunks are read from the data only, as DataArray.chunksizes also
        # checks time coordinates, which may be chunked differently.
        data_chunks = dict(
            zip(correlated_data.dims, correlated_data.chunks or ())
        )
        if "frequency" in data_chunks:
            gains = gains.chunk({"frequency": data_chunks["frequency"]})
    else:  # jones_type == T or G
        assert gains.frequency.size == 1, "Gaintable frequency "
        "must either match to visibility frequency, or must be of size 1"
        # Remove frequency dimension for apply_ufunc to work properly
        gains = gains.isel(frequency=0, drop=True)

    soln_interval_slices = gaintable.soln_interval_slices
    antenna1, antenna2 = get_baseline_antenna_indices(ms_xdt)

    applied: xr.DataArray = xr.concat(
        [
            xr.apply_ufunc(
                _apply_gaintable_to_dataset_ufunc,
                correlated_data.isel(time=slc),
                gains.isel(time=idx, drop=True),
                input_core_dims=[
                    ["baseline_id", "polarization"],
                    ["antenna_name", "polarization"],
                ],
                output_core_dims=[
                    ["baseline_id", "polarization"],
                ],
                dask="parallelized",
                output_dtypes=[correlated_data.dtype],
                kwargs={
                    "antenna1": antenna1,
                    "antenna2": antenna2,
                    "inverse": inverse,
                },
            )
            for idx, slc in enumerate(soln_interval_slices)
        ],
        dim="time",
    )

    return with_correlated_data(ms_xdt, applied)
