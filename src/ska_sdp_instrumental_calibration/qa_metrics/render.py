import html
from importlib.resources import files
from string import Template

TEMPLATES = files(__package__) / "templates"

PAGE_TEMPLATE = Template((TEMPLATES / "qa_metric.html").read_text())


PAGE_STYLE = (TEMPLATES / "qa_metric.css").read_text()


def render_section(title: str) -> str:
    """
    Render a collapsible section .

    Parameters
    ----------
    title : str
        Section header.

    Returns
    -------
    str
        HTML of the section, expanded by default.
    """
    return (
        f"<details open><summary>{html.escape(title)}</summary>"
        "<div></div></details>"
    )


def render_page(title: str, sections: list[str]) -> str:
    """
    Render the QA metric page.

    Parameters
    ----------
    title : str
        Page title.
    sections : list of str
        HTML of the sections, in display order.

    Returns
    -------
    str
        HTML of the page.
    """
    return PAGE_TEMPLATE.substitute(
        title=html.escape(title),
        style=PAGE_STYLE,
        sections="\n".join(sections),
    )
