import logging

from ..data_managers.data_export.qa_metadata import read_qa_metadata_from_dir
from ..data_managers.schema.qa_metadata import QAMetadata
from ..data_managers.schema.qa_metric import (
    BandpassFitMetrics,
    CableDelays,
    FlaggingStatistics,
)
from ..data_managers.sdm import get_final_gaintable_path
from ..exceptions import QAMetadataError
from .report import QAMetricReport

logger = logging.getLogger(__name__)

REPORT_TITLE = "Bandpass QA Metrics"


def generate_qa_metric(qa_dir: str, sdm_path: str | None = None):
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
        qa_metadata = read_qa_metadata_from_dir(qa_dir)
    except QAMetadataError as error:
        logger.warning("%s", error)
        _build_qa_metric_report().write_report_to_dir(qa_dir)
        return

    for metadata in qa_metadata:
        final_gaintable_path = get_final_gaintable_path(
            qa_dir, metadata.source_name, sdm_path
        )
        # Both sections need the final gaintable: read together, so a
        # missing gaintable is logged once
        try:
            flagging_statistics = FlaggingStatistics.read_from_gaintable(
                final_gaintable_path, metadata.pre_flagging
            )
            bandpass_fit_metrics = BandpassFitMetrics.read_from_gaintable(
                final_gaintable_path
            )
        except FileNotFoundError as error:
            logger.warning("%s", error)
            flagging_statistics = bandpass_fit_metrics = None

        _build_qa_metric_report(
            metadata,
            flagging_statistics,
            bandpass_fit_metrics,
            CableDelays.from_qa_metadata(metadata),
        ).write_report_to_dir(qa_dir)


def _build_qa_metric_report(
    metadata: QAMetadata | None = None,
    flagging_statistics: FlaggingStatistics | None = None,
    bandpass_fit_metrics: BandpassFitMetrics | None = None,
    cable_delays: CableDelays | None = None,
) -> QAMetricReport:
    """
    Build the QA metric report of a field.

    Parameters
    ----------
    metadata : QAMetadata, optional
        QA metadata of the field. Without it, every section shows the
        pre-requisite data missing error.
    flagging_statistics : FlaggingStatistics, optional
        Flagging statistics of the field.
    bandpass_fit_metrics : BandpassFitMetrics, optional
        Bandpass fit metrics of the field.
    cable_delays : CableDelays, optional
        Cable delays of the field.

    Returns
    -------
    QAMetricReport
        The report with all its sections.
    """
    return (
        QAMetricReport(REPORT_TITLE, metadata)
        .add_metadata_summary()
        .add_flagging_statistics(flagging_statistics)
        .add_bandpass_fit_metrics(bandpass_fit_metrics)
        .add_cable_delays(cable_delays)
    )
