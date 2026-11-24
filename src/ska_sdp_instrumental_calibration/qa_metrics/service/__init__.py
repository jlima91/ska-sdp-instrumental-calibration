"""
Services of the sections of a field's QA report: the lookups of their
products in the run metadata and the computation of their metrics.

Each module serves one section:

``metadata_summary_metric``
    Looks up the stations of a field, used by every section, and computes
    its observation parameters from the timestamps of its measurement
    sets.
``flagging_statistics_metric``
    Looks up the final gaintable and the pre-flagging data of a field.
``bandpass_fit_metric``
    Looks up the bandpass gaintables of a field and computes the phase and
    gain amplitude fit quality of each of its stations.
``cable_delays_metric``
    Looks up the delay tables of a field.

A lookup returns an empty list where a product with one path per
measurement set is not recorded: its reader raises a ``RunMetadataError``
for it. A missing single product, such as the final gaintable, is raised
by its lookup.
"""

from .bandpass_fit_metric import (
    bandpass_fit_quality,
    get_bandpass_gaintable_paths,
)
from .cable_delays_metric import get_delaytable_paths
from .flagging_statistics_metric import (
    get_final_gaintable,
    get_pre_flagging_paths,
)
from .metadata_summary_metric import (
    get_observation_parameters,
    get_station_configuration,
)

__all__ = [
    "bandpass_fit_quality",
    "get_bandpass_gaintable_paths",
    "get_delaytable_paths",
    "get_final_gaintable",
    "get_observation_parameters",
    "get_pre_flagging_paths",
    "get_station_configuration",
]
