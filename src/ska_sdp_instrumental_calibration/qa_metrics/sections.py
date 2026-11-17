"""
View models of the sections of the QA metric report: the data needed to
draw a section, built by QAMetricReport.
"""

from dataclasses import dataclass, field

from ..data_managers.schema.qa_metadata import QAMetadata
from .tables import Column, ColumnGroup


@dataclass
class Section:
    """
    A collapsible section of the QA metric report.

    Parameters
    ----------
    title : str
        Header of the section.
    template : str
        Template that draws the content of the section.
    error : str, optional
        Error message, shown instead of the content.
    """

    title: str
    template: str
    error: str | None = None


@dataclass
class MetadataSection(Section):
    """
    Metadata summary of a field.

    Parameters
    ----------
    metadata : QAMetadata, optional
        QA metadata of the field.
    """

    metadata: QAMetadata | None = None


@dataclass
class TableSection(Section):
    """
    A section with a sortable, paginated table.

    Parameters
    ----------
    columns : list[Column | ColumnGroup]
        Columns of the table.
    rows : list[dict], optional
        Rows of the table.
    item_name : str
        Name of the rows, shown in the pagination, e.g. ``stations``.
    """

    columns: list[Column | ColumnGroup] = field(default_factory=list)
    rows: list[dict] | None = None
    item_name: str = "stations"


@dataclass
class FlaggingStatisticsSection(TableSection):
    """
    Flagging statistics table, with the totals over all stations.

    Parameters
    ----------
    totals : dict[str, float | None], optional
        ``flagged_before`` and ``flagged_after`` average percentage.
    """

    totals: dict[str, float | None] | None = None
