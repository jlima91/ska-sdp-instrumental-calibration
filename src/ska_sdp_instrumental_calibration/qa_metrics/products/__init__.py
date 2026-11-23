"""
Products of a run read or written for the QA report.

Each module handles one product:

``gaintable``
    Reads the gaintables written by the pipeline, e.g. the final gaintable
    of a field, as a ``GainCalibrationSetXds``.
``delaytable``
    Reads the delay tables written by the delay stage, one per measurement
    set, as one ``DelayTable`` per field.
``pre_flagging``
    Writes and reads the pre-flagging data of each measurement set, kept in
    the ``derived/`` folder. Its layout is owned here, by its writer and
    reader.
``gain_fits``
    Writes the curve fits of gain flagging to the ``derived/`` folder, for
    the QA plots.

Readers take every path of a product for a field and return it as one
object, concatenated along time where a field has several. Every missing
or unreadable product is raised as a ``RunMetadataError``, which the QA
report shows in the section that needs it.
"""

from .delaytable import read_delaytable
from .gain_fits import export_gain_fits_to_zarr
from .gaintable import read_gaintable
from .pre_flagging import export_pre_flagging, read_pre_flagging

__all__ = [
    "export_gain_fits_to_zarr",
    "export_pre_flagging",
    "read_delaytable",
    "read_gaintable",
    "read_pre_flagging",
]
