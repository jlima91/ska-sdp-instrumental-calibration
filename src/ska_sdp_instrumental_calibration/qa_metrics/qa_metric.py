import logging
import os

from ..data_managers.sdm import get_qa_metric_dir, get_qa_metric_field_id
from .render import render_page, render_section

logger = logging.getLogger(__name__)

PAGE_TITLE = "Bandpass QA Metrics"

SECTION_TITLES = [
    "Metadata",
    "Antenna flagging table",
    "Bandpass fit metrics table",
    "Bandpass Plots",
    "Cable Delays table",
]


def generate_qa_metric(output_path: str, sdm_path: str | None = None):
    """
    Write the QA metric HTML page for bandpass stage of the INST run,
    as ``inst-<field-id>-qa-metric.html``.

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
    qa_metric_dir = get_qa_metric_dir(output_path, sdm_path)
    page = render_page(
        PAGE_TITLE, [render_section(title) for title in SECTION_TITLES]
    )

    field_id = get_qa_metric_field_id(qa_metric_dir, sdm_path)
    write_qa_metric(page, qa_metric_dir, field_id)


def write_qa_metric(page: str, qa_metric_dir: str, field_id: str):
    """
    Write a QA metric HTML page.

    Parameters
    ----------
    page : str
        HTML of the page.
    qa_metric_dir : str
        Directory to write the page to.
    field_id : str
        Field ID the page belongs to.
    """
    qa_metric_path = os.path.join(
        qa_metric_dir, f"inst-{field_id}-qa-metric.html"
    )
    with open(qa_metric_path, "w", encoding="utf-8") as qa_metric_file:
        qa_metric_file.write(page)

    logger.info("QA metric written to %s", qa_metric_path)
