import logging
import os

import jinja2

from ..data_managers.schema.qa_metadata import QAMetadata
from ..data_managers.schema.qa_metric import (
    BandpassFitMetrics,
    CableDelays,
    FlaggingStatistics,
)
from .sections import (
    FlaggingStatisticsSection,
    MetadataSection,
    Section,
    TableSection,
)
from .tables import (
    BANDPASS_FIT_METRICS_COLUMNS,
    CABLE_DELAYS_COLUMNS,
    FLAGGING_STATISTICS_COLUMNS,
    build_station_rows,
)

logger = logging.getLogger(__name__)

GAINTABLE_MISSING_ERROR = "ERROR: Gaintable missing"
DELAY_TABLE_MISSING_ERROR = "ERROR: Delay table missing"
PREREQUISITE_DATA_MISSING_ERROR = "ERROR: Pre-requisite data missing"

PLACEHOLDER_SOURCE_NAME = "PLACEHOLDER_SOURCE_NAME"

template_env = jinja2.Environment(
    loader=jinja2.PackageLoader(__package__),
    autoescape=jinja2.select_autoescape(),
)

SectionData = FlaggingStatistics | BandpassFitMetrics | CableDelays


class QAMetricReport:
    """
    Builder of the QA metric HTML report of one field. Sections are shown
    in the order they are added.

    A section whose input is missing shows an error message instead of its
    content.

    Parameters
    ----------
    title : str
        Page title.
    metadata : QAMetadata, optional
        QA metadata of the field. Without it, every section shows the
        pre-requisite data missing error.
    """

    def __init__(self, title: str, metadata: QAMetadata | None = None):
        self.title = title
        self.metadata = metadata
        self.sections: list[Section] = []

    def add_metadata_summary(self) -> "QAMetricReport":
        """
        Add the metadata summary section.

        Returns
        -------
        QAMetricReport
            The report, to chain the next section.
        """
        self.sections.append(
            MetadataSection(
                "Metadata summary",
                "sections/metadata.html",
                error=self._section_error(
                    self.metadata, PREREQUISITE_DATA_MISSING_ERROR
                ),
                metadata=self.metadata,
            )
        )
        return self

    def add_flagging_statistics(
        self, flagging_statistics: FlaggingStatistics | None = None
    ) -> "QAMetricReport":
        """
        Add the flagging statistics section: flagged percentage per station
        before and after calibration, and the totals over all stations.

        Parameters
        ----------
        flagging_statistics : FlaggingStatistics, optional
            Flagging statistics of the field. None if its final gaintable
            is missing.

        Returns
        -------
        QAMetricReport
            The report, to chain the next section.
        """
        section = FlaggingStatisticsSection(
            "Flagging Statistics",
            "sections/flagging_statistics.html",
            columns=FLAGGING_STATISTICS_COLUMNS,
        )
        if flagging_statistics is not None:
            section.totals = flagging_statistics.totals()

        return self._add_station_table(
            section, flagging_statistics, GAINTABLE_MISSING_ERROR
        )

    def add_bandpass_fit_metrics(
        self, bandpass_fit_metrics: BandpassFitMetrics | None = None
    ) -> "QAMetricReport":
        """
        Add the bandpass fit metrics section, one row per station.

        Parameters
        ----------
        bandpass_fit_metrics : BandpassFitMetrics, optional
            Bandpass fit metrics of the field. None if its final gaintable
            is missing.

        Returns
        -------
        QAMetricReport
            The report, to chain the next section.
        """
        return self._add_station_table(
            TableSection(
                "Bandpass Fit Metrics",
                "sections/table.html",
                columns=BANDPASS_FIT_METRICS_COLUMNS,
                item_name="antennas",
            ),
            bandpass_fit_metrics,
            GAINTABLE_MISSING_ERROR,
        )

    def add_cable_delays(
        self, cable_delays: CableDelays | None = None
    ) -> "QAMetricReport":
        """
        Add the cable delays section, one row per station.

        Parameters
        ----------
        cable_delays : CableDelays, optional
            Cable delays of the field. None if its delay table is missing.

        Returns
        -------
        QAMetricReport
            The report, to chain the next section.
        """
        return self._add_station_table(
            TableSection(
                "Cable Delays",
                "sections/table.html",
                columns=CABLE_DELAYS_COLUMNS,
                item_name="antennas",
            ),
            cable_delays,
            DELAY_TABLE_MISSING_ERROR,
        )

    def render_html(self) -> str:
        """
        Render the report.

        Returns
        -------
        str
            HTML of the page.
        """
        template = template_env.get_template("qa_metric.html")
        return template.render(
            title=self.title,
            metadata=self.metadata,
            sections=self.sections,
        )

    def write_report_to_dir(self, output_dir: str):
        """
        Write the report to ``<output_dir>/inst-<source-name>-qa-metric.html``.

        Parameters
        ----------
        output_dir : str
            Directory to write the report to.
        """
        source_name = (
            self.metadata.source_name
            if self.metadata
            else PLACEHOLDER_SOURCE_NAME
        )
        qa_metric_path = os.path.join(
            output_dir, f"inst-{source_name}-qa-metric.html"
        )
        with open(qa_metric_path, "w", encoding="utf-8") as qa_metric_file:
            qa_metric_file.write(self.render_html())

        logger.info("QA metric written to %s", qa_metric_path)

    def _add_station_table(
        self,
        section: TableSection,
        section_data: SectionData | None,
        missing_error: str,
    ) -> "QAMetricReport":
        """
        Add a table section with one row per station, or its error if its
        data is missing.
        """
        section.error = self._section_error(section_data, missing_error)
        if section.error is None:
            section.rows = build_station_rows(
                self.metadata.station_configuration,
                section_data.values_per_station(),
            )

        self.sections.append(section)
        return self

    def _section_error(
        self, section_data: object | None, missing_error: str
    ) -> str | None:
        """
        Error of a section: pre-requisite data missing without the QA
        metadata, missing_error without the section data, else None.
        """
        if self.metadata is None:
            return PREREQUISITE_DATA_MISSING_ERROR

        return missing_error if section_data is None else None
