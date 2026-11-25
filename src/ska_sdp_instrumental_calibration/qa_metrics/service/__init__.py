"""
Services of the sections of a field's QA report: the lookups of their
products in the run metadata and the computation of their metrics.

Each module serves one section:

``metadata_summary``
    Looks up the stations of a field, used by every section, its file
    information, and computes its observation parameters from the
    timestamps of its measurement sets.
``flagging_statistics``
    Looks up the final gaintable of a field and computes its flagging
    statistics from it.
``pre_flagging``
    Writes, looks up and reads the pre-flagging data of a field.
``bandpass_fit``
    Looks up the bandpass gaintables of a field and computes the phase and
    gain amplitude fit quality of each of its stations.
``cable_delays``
    Looks up and reads the delay tables of a field and computes its cable
    delays.

A lookup returns an empty list where a product with one path per
measurement set is not recorded: its reader raises a ``RunMetadataError``
for it. A missing single product, such as the final gaintable, is raised
by its lookup.
"""

from .bandpass_fit import bandpass_fit_quality, get_bandpass_gaintable_paths
from .cable_delays import get_delaytable_paths, read_delaytable
from .flagging_statistics import get_final_gaintable
from .gain_fits import export_gain_fits_to_zarr
from .gaintable import read_gaintable
from .metadata_summary import (
    get_observation_parameters,
    get_station_configuration,
)
from .pre_flagging import (
    export_pre_flagging,
    get_pre_flagging_paths,
    read_pre_flagging,
)

__all__ = [
    "bandpass_fit_quality",
    "get_bandpass_gaintable_paths",
    "get_delaytable_paths",
    "get_final_gaintable",
    "get_observation_parameters",
    "get_pre_flagging_paths",
    "get_station_configuration",
    "export_gain_fits_to_zarr",
    "export_pre_flagging",
    "read_delaytable",
    "read_gaintable",
    "read_pre_flagging",
]
