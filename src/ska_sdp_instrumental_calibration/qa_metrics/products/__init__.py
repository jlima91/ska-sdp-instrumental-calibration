"""
Products of a run read or written for the QA report: the gaintables and
the derived data kept in the ``derived/`` folder. Each module owns the
layout of one product, with both its writer and its reader.
"""

from .gain_fits import export_gain_fits_to_zarr
from .gaintable import read_gaintable
from .pre_flagging import export_pre_flagging, read_pre_flagging

__all__ = [
    "export_gain_fits_to_zarr",
    "export_pre_flagging",
    "read_gaintable",
    "read_pre_flagging",
]
