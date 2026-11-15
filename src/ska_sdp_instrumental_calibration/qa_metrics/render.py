import jinja2

from .tables import FLAGGING_STATISTICS_COLUMNS

env = jinja2.Environment(
    loader=jinja2.PackageLoader(__package__),
    autoescape=jinja2.select_autoescape(),
)


def render_page(
    title: str,
    section_titles: list[str],
    metadata: dict | None = None,
    flagging_statistics: list[dict] | None = None,
    section_errors: dict[str, str] | None = None,
    flagging_totals: dict[str, float | None] | None = None,
) -> str:
    """
    Render the QA metric page.

    Parameters
    ----------
    title : str
        Page title.
    section_titles : list of str
        Headers of the sections, in display order.
    metadata : dict, optional
        QA metadata of the INST run, to fill the metadata section.
    flagging_statistics : list of dict, optional
        Rows of the flagging statistics table.
    section_errors : dict of str, optional
        Error message keyed by section (``metadata``,
        ``flagging_statistics``), shown instead of the section content.
    flagging_totals : dict, optional
        Flagged percentage over all stations, before and after.

    Returns
    -------
    str
        HTML of the page.
    """
    template = env.get_template("qa_metric.html")
    return template.render(
        title=title,
        section_titles=section_titles,
        metadata=metadata,
        flagging_statistics=flagging_statistics,
        flagging_statistics_columns=FLAGGING_STATISTICS_COLUMNS,
        section_errors=section_errors or {},
        flagging_totals=flagging_totals,
    )
