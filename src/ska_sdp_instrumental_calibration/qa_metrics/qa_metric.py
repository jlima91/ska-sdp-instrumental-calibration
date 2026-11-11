import logging
import os

from ska_sdp_piper.piper.log import PiperLogging

from ..data_managers.data_export.qa_metadata import (
    get_qa_metadata_paths,
    read_qa_metadata,
)
from ..data_managers.sdm import get_final_gaintable_path, get_qa_metric_dir
from .flagging_statistics import (
    build_flagging_rows,
    gaintable_flagged_percentage,
)
from .formatting import format_flagging_statistics, format_metadata
from .render import render_page

logger = logging.getLogger(__name__)

PAGE_TITLE = "Bandpass QA Metrics"

SECTION_TITLES = [
    "Metadata summary",
    "Flagging Statistics",
    "Bandpass fit metrics table",
    "Bandpass Plots",
    "Cable Delays table",
]

GAINTABLE_MISSING = "ERROR: Gaintable missing"
PREREQUISITE_DATA_MISSING = "ERROR: Pre-requisite data missing"


def generate_qa_metric(output_path: str, sdm_path: str | None = None):
    """
    Write the QA metric HTML page for bandpass stage of the INST run,
    as ``inst-<source-name>-qa-metric.html``.

    The report is stored in the logs directory of the latest INST run if an SDM
    path is provided, else to ``output_path`` provided.

    Parameters
    ----------
    output_path : str
        Output folder of the INST run. Used if no SDM path is provided.
    sdm_path : str, optional
        Path to the SDM directory. Takes precedence over ``output_path``
        if both are given
    """
    PiperLogging.configure()

    qa_metric_dir = get_qa_metric_dir(output_path, sdm_path)

    qa_metadata_paths = get_qa_metadata_paths(qa_metric_dir)
    if not qa_metadata_paths:
        logger.warning("No QA metadata found in %s", qa_metric_dir)
        page = render_page(
            PAGE_TITLE,
            SECTION_TITLES,
            section_errors={
                "metadata": PREREQUISITE_DATA_MISSING,
                "flagging_statistics": PREREQUISITE_DATA_MISSING,
            },
        )
        write_qa_metric(page, qa_metric_dir, "PLACEHOLDER_SOURCE_NAME")
        return

    for qa_metadata_path in qa_metadata_paths:
        metadata = read_qa_metadata(qa_metadata_path)
        flagging_statistics, flagging_error = _get_flagging_statistics(
            metadata, qa_metric_dir, sdm_path
        )

        page = render_page(
            PAGE_TITLE,
            SECTION_TITLES,
            format_metadata(metadata),
            flagging_statistics,
            {"flagging_statistics": flagging_error},
        )
        write_qa_metric(page, qa_metric_dir, metadata["source_name"])


def _get_flagging_statistics(
    metadata: dict, qa_metric_dir: str, sdm_path: str | None
) -> tuple[list[dict] | None, str | None]:
    """
    Rows of the flagging statistics table of a field, or the error to show
    if its final gaintable is missing.

    Parameters
    ----------
    metadata : dict
        QA metadata of the field.
    qa_metric_dir : str
        Directory of the INST run.
    sdm_path : str, optional
        Path to the SDM directory, if the INST run is in an SDM.

    Returns
    -------
    tuple
        ``(rows, None)``, or ``(None, error message)``.
    """
    final_gaintable_path = get_final_gaintable_path(
        qa_metric_dir, metadata["source_name"], sdm_path
    )
    if not os.path.exists(final_gaintable_path):
        logger.warning("No final gaintable found at %s", final_gaintable_path)
        return None, GAINTABLE_MISSING

    rows = build_flagging_rows(
        metadata["station_configuration"],
        metadata["pre_flagged_gains"],
        gaintable_flagged_percentage(final_gaintable_path),
    )
    return format_flagging_statistics(rows), None


def write_qa_metric(page: str, qa_metric_dir: str, source_name: str):
    """
    Write a QA metric HTML page.

    Parameters
    ----------
    page : str
        HTML of the page.
    qa_metric_dir : str
        Directory to write the page to.
    source_name : str
        Source name the page belongs to.
    """
    qa_metric_path = os.path.join(
        qa_metric_dir, f"inst-{source_name}-qa-metric.html"
    )
    with open(qa_metric_path, "w", encoding="utf-8") as qa_metric_file:
        qa_metric_file.write(page)

    logger.info("QA metric written to %s", qa_metric_path)
