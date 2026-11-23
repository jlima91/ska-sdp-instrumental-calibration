import logging
import os

import jinja2

from .sections import Section

logger = logging.getLogger(__name__)

GAINTABLE_MISSING_ERROR = "ERROR: Gaintable missing"
DELAY_TABLE_MISSING_ERROR = "ERROR: Delay table missing"
PREREQUISITE_DATA_MISSING_ERROR = "ERROR: Pre-requisite data missing"

PLACEHOLDER_SOURCE_NAME = "PLACEHOLDER_SOURCE_NAME"

template_env = jinja2.Environment(
    loader=jinja2.PackageLoader(__package__),
    autoescape=jinja2.select_autoescape(),
)


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

    def __init__(self, title: str):
        self.title = title
        self.sections: list[Section] = []

        self.field_name = PLACEHOLDER_SOURCE_NAME

    def add_field_name(self, field_name: str):
        self.field_name = field_name
        return self

    def add_section(self, section_data: Section):
        self.sections.append(section_data)
        return self

    # def add_bandpass_fit_metrics(
    #     self, bandpass_fit_metrics: BandpassFitMetrics | None = None
    # ) -> "QAMetricReport":
    #     """
    #     Add the bandpass fit metrics section, one row per station.

    #     Parameters
    #     ----------
    #     bandpass_fit_metrics : BandpassFitMetrics, optional
    #         Bandpass fit metrics of the field. None if its final gaintable
    #         is missing.

    #     Returns
    #     -------
    #     QAMetricReport
    #         The report, to chain the next section.
    #     """
    #     return self._add_station_table(
    #         TableSection(
    #             "Bandpass Fit Metrics",
    #             column_group=BANDPASS_FIT_METRICS_COLUMNS,
    #             item_name="antennas",
    #         ),
    #         bandpass_fit_metrics,
    #         GAINTABLE_MISSING_ERROR,
    #     )

    # def add_cable_delays(
    #     self, cable_delays: CableDelays | None = None
    # ) -> "QAMetricReport":
    #     """
    #     Add the cable delays section, one row per station.

    #     Parameters
    #     ----------
    #     cable_delays : CableDelays, optional
    #         Cable delays of the field. None if its delay table is missing.

    #     Returns
    #     -------
    #     QAMetricReport
    #         The report, to chain the next section.
    #     """
    #     return self._add_station_table(
    #         TableSection(
    #             "Cable Delays",
    #             "sections/table.html",
    #             columns=CABLE_DELAYS_COLUMNS,
    #             item_name="antennas",
    #         ),
    #         cable_delays,
    #         DELAY_TABLE_MISSING_ERROR,
    #     )

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
            field_name=self.field_name,
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
        qa_metric_path = os.path.join(
            output_dir, f"inst-{self.field_name}-qa-metric.html"
        )
        with open(qa_metric_path, "w", encoding="utf-8") as qa_metric_file:
            qa_metric_file.write(self.render_html())

        logger.info("QA metric written to %s", qa_metric_path)
