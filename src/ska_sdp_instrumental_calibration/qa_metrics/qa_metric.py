import logging

from . import field_sections
from .exceptions import QAReportException, RunMetadataError
from .report import QAMetricReport
from .run_metadata import RunMetadata
from .sections import ErrorSection, FlaggingStatisticsSection, MetadataSection

logger = logging.getLogger(__name__)


def generate_qa_metric(qa_dir: str, title: str = "Bandpass QA Metrics"):
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
    report_sections = [MetadataSection, FlaggingStatisticsSection]
    try:
        run_metadata = RunMetadata.initialize(qa_dir)
    except RunMetadataError as error:
        logger.warning("%s", error)

        report_error = QAReportException(
            f"Run metadata of the INST run in {qa_dir} cannot be loaded",
            cause=error,
        )

        QAMetricReport(title).add_section(
            ErrorSection("Run metadata", report_error)
        ).write_report_to_dir(qa_dir)

        return

    for field_metadata in run_metadata:
        report = QAMetricReport(title).add_field_name(
            field_metadata.field_name
        )
        for section in report_sections:
            report.add_section(field_sections.build(section, field_metadata))

        report.write_report_to_dir(qa_dir)
