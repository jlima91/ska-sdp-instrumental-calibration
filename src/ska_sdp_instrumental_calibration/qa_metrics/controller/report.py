import logging

from ..exceptions import QAReportException, QAValidationError, RunMetadataError
from ..repository.run_manifest import RunManifest
from ..view.qa_metric_report import QAMetricReport
from ..view.sections import (
    BandpassFitSection,
    CableDelaysSection,
    ErrorSection,
    FlaggingStatisticsSection,
    MetadataSection,
)
from .sections import section_registry

logger = logging.getLogger(__name__)

# Sections of the report of each field, in the order they are shown
REPORT_SECTIONS = (
    MetadataSection,
    FlaggingStatisticsSection,
    BandpassFitSection,
    CableDelaysSection,
)


def generate_qa_metric(
    qa_dir: str, title: str = "Bandpass QA Metrics", **kwargs
):
    """
    Write the QA metric HTML reports of an INST run, one per field, from
    its run metadata.

    Each field report, ``<qa_dir>/inst-<field-name>-qa-metric.html``, has
    the sections of ``REPORT_SECTIONS``, in that order. A section whose
    products are missing or unreadable shows its error instead of its
    content.

    If the run metadata cannot be loaded or records no field, a single
    error report of the run, ``<qa_dir>/inst-qa-metric.html``, is written
    instead. If a field report fails with an unexpected error, i.e. a bug,
    the error is logged with its traceback, no report is written for that
    field, and the other fields are still reported.

    Parameters
    ----------
    qa_dir : str
        QA directory of the INST run, with its ``run_metadata.json``: the
        logs directory of the latest INST run in the SDM, or the output
        directory. The reports are written to it.
    title : str, optional
        Page title of the reports.
    """
    exit_status: dict[str, list] = {}
    try:
        run_metadata = RunManifest.initialize(qa_dir)
        if not run_metadata.field_metadata:
            raise RunMetadataError(
                f"No field recorded in the run metadata of {qa_dir}"
            )

    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.warning(
            "%s", error, exc_info=not isinstance(error, RunMetadataError)
        )

        report_error = QAReportException(
            f"QA metric report of the INST run in {qa_dir} cannot be built",
            cause=error,
        )

        QAMetricReport(title).add_section(
            ErrorSection("Run metadata", report_error)
        ).write_report_to_dir(qa_dir)

        return

    for field_metadata in run_metadata:
        try:
            report = QAMetricReport(title).add_field_name(
                field_metadata.field_name
            )
            for section in REPORT_SECTIONS:
                report_section = section_registry.build(
                    section, field_metadata, **kwargs
                )
                report.add_section(report_section)
                if report_section.exit_status is not None:
                    exit_status.setdefault(
                        field_metadata.field_name, []
                    ).append(report_section.exit_status)

            report.write_report_to_dir(qa_dir)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception(
                "QA metric report of field %s cannot be written",
                field_metadata.field_name,
            )

    if exit_status:
        messages = ""
        for field_name, statuses in exit_status.items():
            reason = "; ".join([status for status in statuses])
            messages += (
                "\nQA metric report of field %s failed with exit status %s"
                % (field_name, reason)
            )
            logger.error(
                "QA metric report of field %s failed with exit status %s",
                field_name,
                reason,
            )
            logger.error(messages)
        raise QAValidationError(messages)
