import glob
import json
import logging
import os
from typing import Callable

import numpy as np

from ...xarray_processors.vis_flagging import (
    flagged_sample_counts,
    flagged_time_count,
    unsolvable_gain_counts,
)

logger = logging.getLogger(__name__)

QA_METADATA_FILE_NAME = "qa-metadata-data.json"


def get_ms_qa_data(vis, input_ms: str) -> dict:
    """
    Get the visibility data of one measurement set needed for the QA
    metadata.

    Parameters
    ----------
    vis: Visibility
        Visibility dataset of the measurement set.
    input_ms: str
        Path of the measurement set.

    Returns
    -------
    dict
        QA data of the measurement set.
    """
    return {
        "input_ms": input_ms,
        "timestamps": vis.time.data,
        "channels": vis.sizes["frequency"],
        "station_names": vis.attrs["configuration"].names.data,
        "antenna1": vis.antenna1.data,
        "antenna2": vis.antenna2.data,
        "flagged_time_count": flagged_time_count(vis).data,
        "ntimes": vis.sizes["time"],
    }


def _percentage_per_station(
    count_function: Callable, ms_qa_data: list[dict], nstations: int
) -> np.ndarray:
    """
    Sum the (flagged, total) counts over the measurement
    sets, then give the percentage per station.
    """
    flagged = 0
    total = 0
    for ms in ms_qa_data:
        ms_flagged, ms_total = count_function(
            ms["flagged_time_count"],
            ms["antenna1"],
            ms["antenna2"],
            ms["ntimes"],
            nstations,
        )
        flagged = flagged + ms_flagged
        total = total + ms_total

    return flagged / total * 100


# TODO: need to get the logic checked from domian exp
def write_qa_metadata(field_id: str, ms_qa_data: list[dict], qa_dir: str):
    """
    Write the QA metadata of the measurement sets of one field to
    <qa_dir>/<field_id>_qa-metadata-data.json. The measurement sets
    are joined along time, like the gaintable.

    Parameters
    ----------
    field_id: str
        Field name. Saved as the source name.
    ms_qa_data: list[dict]
        QA data of each measurement set of the field, from get_ms_qa_data.
    qa_dir: str
        QA directory of the run: the SDM logs directory or the output
        directory. Saved as the output path.
    """
    # Measurement sets of a field share stations and channels, so they
    # are taken from the first one
    first_ms = ms_qa_data[0]
    station_names = [str(name) for name in first_ms["station_names"]]
    nstations = len(station_names)

    # Integration time is the time resolution: the spacing of the MS TIME
    # column, as it follows any time averaging of the data. The median
    # ignores gaps from missing time steps. It is taken within the first
    # measurement set, so the gap between measurement sets is not counted
    integration_time = np.median(np.diff(first_ms["timestamps"]))

    # MS TIME is the centre of each integration, so the observation starts
    # half an integration before the first time
    start_time = (
        min(ms["timestamps"].min() for ms in ms_qa_data) - integration_time / 2
    )

    # Flagged visibility samples, as asked by the AC
    pre_flagged_data = _percentage_per_station(
        flagged_sample_counts, ms_qa_data, nstations
    )
    # Gains that cannot be solved, in the same unit as the flagged gains
    # of the final gaintable, for the flagging statistics table
    pre_flagged_gains = _percentage_per_station(
        unsolvable_gain_counts, ms_qa_data, nstations
    )

    qa_metadata = {
        "source_name": field_id,
        "input_ms": [ms["input_ms"] for ms in ms_qa_data],
        "output_path": qa_dir,
        "observation_parameters": {
            "start_time": float(start_time),
            "integration_time": float(integration_time),
            "channels": int(first_ms["channels"]),
        },
        "pre_flagged_data": {
            name: float(percentage)
            for name, percentage in zip(station_names, pre_flagged_data)
        },
        "pre_flagged_gains": {
            name: float(percentage)
            for name, percentage in zip(station_names, pre_flagged_gains)
        },
        "station_configuration": {
            str(station_id): name
            for station_id, name in enumerate(station_names)
        },
    }

    qa_metadata_path = os.path.join(
        qa_dir, f"{field_id}_{QA_METADATA_FILE_NAME}"
    )
    with open(qa_metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(qa_metadata, metadata_file, indent=4)

    logger.info("QA metadata written to %s", qa_metadata_path)


def get_qa_metadata_paths(qa_metric_dir: str) -> list[str]:
    """
    Get the QA metadata JSON files of an INST run, one per field.

    Parameters
    ----------
    qa_metric_dir: str
        Directory of the INST run.

    Returns
    -------
    list[str]
        Paths of the <field>_qa-metadata-data.json file in
        qa_metric_dir
    """
    return sorted(
        glob.glob(os.path.join(qa_metric_dir, f"*_{QA_METADATA_FILE_NAME}"))
    )


def read_qa_metadata(qa_metadata_path: str) -> dict:
    """
    Read a QA metadata JSON file.

    Parameters
    ----------
    qa_metadata_path: str
        Path of a QA metadata JSON file.

    Returns
    -------
    dict
        The QA metadata.
    """
    with open(qa_metadata_path, encoding="utf-8") as metadata_file:
        return json.load(metadata_file)
