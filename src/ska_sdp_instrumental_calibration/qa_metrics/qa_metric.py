import glob
import logging
import os

from ..data_managers.sdm import get_final_gaintable_path
from ..exceptions import QAMetadataError
from .domain.qa_metadata import QA_METADATA_FILE_NAME, QAMetadata
from .domain.qa_metric import (
    BandpassFitMetrics,
    CableDelays,
    FlaggingStatistics,
)
from .field_queries import (
    get_bandpass_fit_metrics,
    get_cable_delays,
    get_flagging_statistics,
    get_qa_metadata,
)
from .qa_section import QASection
from .report import QAMetricReport
from .run_metadata import RunMetadata

logger = logging.getLogger(__name__)

REPORT_TITLE = "Bandpass QA Metrics"


def generate_qa_metric(qa_dir: str):
    """
    Write the QA metric HTML report for bandpass stage of the INST run, one
    per source, as ``<qa_dir>/inst-<source-name>-qa-metric.html``.

    Parameters
    ----------
    qa_dir : str
        QA directory of the INST run: the logs directory of the latest INST
        run in the SDM, or the output directory.
    sdm_path : str, optional
        Path to the SDM directory, if the INST run is in an SDM.
    """
    try:
        run_metadata = RunMetadata.initialize(qa_dir)
    except QAMetadataError as error:
        logger.warning("%s", error)
        QAMetricReport(REPORT_TITLE, None).add_metadata_summary(
            None
        ).write_report_to_dir(qa_dir)

        return

    for field_metadata in run_metadata:
        qa_metadata = get_qa_metadata(field_metadata)
        # flagging_statistics = get_flagging_statistics(field_metadata)
        # bandpass_fit_metrics = get_bandpass_fit_metrics(field_metadata)
        # cable_delays = get_cable_delays(field_metadata)
        (
            QAMetricReport(REPORT_TITLE, qa_metadata).add_metadata_summary(
                qa_metadata
            )
            # .add_flagging_statistics(flagging_statistics)
            # .add_bandpass_fit_metrics(bandpass_fit_metrics)
            # .add_cable_delays(cable_delays)
        ).write_report_to_dir(qa_dir)
