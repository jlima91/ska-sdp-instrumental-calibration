from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader(__package__), autoescape=select_autoescape()
)


def render_page(title: str, section_titles: list[str]) -> str:
    """
    Render the QA metric page.

    Parameters
    ----------
    title : str
        Page title.
    section_titles : list of str
        Headers of the sections, in display order.

    Returns
    -------
    str
        HTML of the page.
    """
    template = env.get_template("qa_metric.html")
    return template.render(title=title, section_titles=section_titles)
