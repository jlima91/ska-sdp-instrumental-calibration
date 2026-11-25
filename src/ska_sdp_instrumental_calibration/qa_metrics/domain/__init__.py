"""
Domain objects of the QA report: the metadata and metrics of a field,
computed from its products.

They are plain data, built from arrays, counts and xarray tables already
read by :mod:`..service`, so they read no files and raise no run metadata
errors. The sections of the report are built from them.

``metadata``
    Metadata of a field: its files, observation parameters and stations.
``flagging_statistics``
    Flagging of a field before calibration (pre-flagging) and its
    flagging statistics after it.
``bandpass_fit``
    Fit metrics of the bandpass gains of a field, per station.
``cable_delays``
    X and Y cable delays of a field, per station.
"""

from .flagging_statistics import FlaggingStatistics, PreFlagging
from .metadata import ObservationParameters, QAMetadata

__all__ = [
    "ObservationParameters",
    "PreFlagging",
    "QAMetadata",
    "FlaggingStatistics",
]
