"""
Gain fits: the curve fits produced by gain flagging, kept in the
``derived/`` folder as a Zarr store for the QA plots.
"""

import logging
from typing import Optional

import xarray as xr

logger = logging.getLogger(__name__)


def export_gain_fits_to_zarr(
    fits: dict[str, xr.DataArray],
    filename: str,
    attrs: Optional[dict] = None,
):
    """
    Export the curve fits produced by gain flagging to a Zarr store.

    The fits are combined into a single xarray Dataset, with one data
    variable per fit type (e.g. ``amp_fit``, ``phase_fit``), so that a
    separate diagnostic process can read them back with
    ``xr.open_zarr(filename)``.

    Parameters
    ----------
    fits : dict[str, xr.DataArray]
        Mapping from fit name to DataArray holding the fitted values.
        All DataArrays must share the same dims and coords.
    filename : str
        Path of the Zarr store to write. Any existing store is overwritten.
    attrs : dict, optional
        Attributes to attach to the Dataset. Zarr stores attrs as JSON, so
        values must be JSON-serialisable.
    """
    fits_ds = xr.Dataset(fits)
    if attrs:
        fits_ds = fits_ds.assign_attrs(attrs)

    logger.info("Writing gain flagging fits to %s", filename)
    fits_ds.to_zarr(filename, mode="w")
