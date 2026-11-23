"""
Domain objects of the QA report: the metadata and metrics of a field,
computed from its products.

They are plain data, built from arrays, counts and xarray tables already
read by :mod:`..products`, so they read no files and raise no run metadata
errors. The sections of the report are built from them.

``qa_metadata``
    Metadata of a field: its files, observation parameters, stations and
    flagging before calibration.
``qa_metric``
    Metrics of a field: flagging statistics, bandpass fit metrics and
    cable delays.
"""

from .qa_metadata import ObservationParameters, PreFlagging, QAMetadata

__all__ = [
    "ObservationParameters",
    "PreFlagging",
    "QAMetadata",
]
