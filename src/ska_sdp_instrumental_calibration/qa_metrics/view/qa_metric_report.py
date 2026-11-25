import logging
import os

import jinja2

from .sections import Section

logger = logging.getLogger(__name__)

# Templates live in the templates folder of this view package. Its name is
# taken from __name__, a str, rather than __package__, typed str | None
template_env = jinja2.Environment(
    loader=jinja2.PackageLoader(__name__.rpartition(".")[0]),
    autoescape=jinja2.select_autoescape(),
)


class QAMetricReport:
    """
    Builder of the QA metric HTML report of one field. Sections are shown
    in the order they are added.

    A section whose input is missing shows an error message instead of its
    content.

    Without a field name, e.g. for the error report of a run whose fields
    cannot be reported, the report is of the whole run: it shows no source
    and is written without a field name in its file name.

    Parameters
    ----------
    title : str
        Page title.
    """

    def __init__(self, title: str):
        self.title = title
        self.sections: list[Section] = []

        # None rather than a placeholder name, which a real field could
        # have: the report of the run then cannot overwrite a field report
        self.field_name: str | None = None

    def add_field_name(self, field_name: str):
        self.field_name = field_name
        return self

    def add_section(self, section_data: Section):
        self.sections.append(section_data)
        return self

    def write_report_to_dir(self, output_dir: str):
        """
        Write the report to ``<output_dir>/inst-<field-name>-qa-metric.html``,
        or ``<output_dir>/inst-qa-metric.html`` without a field name.

        Parameters
        ----------
        output_dir : str
            Directory to write the report to.
        """
        file_name = (
            "inst-qa-metric.html"
            if self.field_name is None
            else f"inst-{self.field_name}-qa-metric.html"
        )
        qa_metric_path = os.path.join(output_dir, file_name)
        template = template_env.get_template("qa_metric.html")
        html_report = template.render(
            title=self.title,
            field_name=self.field_name,
            sections=self.sections,
        )
        with open(qa_metric_path, "w", encoding="utf-8") as qa_metric_file:
            qa_metric_file.write(html_report)

        logger.info("QA metric written to %s", qa_metric_path)
