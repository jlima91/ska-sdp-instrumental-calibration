"""
View models of the sections of the QA metric report: the data needed to
draw a section, built by QAMetricReport.

Each section type draws with its own template, a class variable, so it is
never passed in. Sections with a fixed title set it as a default left out of
``__init__``; the others take it as their first argument.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from .domain.qa_metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from .exceptions import QAReportException
from .tables import Column, ColumnGroup


@dataclass
class Section:
    """
    A collapsible section of the QA metric report.

    Attributes
    ----------
    title : str
        Header of the section. Set by each subclass, either as a fixed
        default or as an ``__init__`` argument.
    template : str
        Template that draws the content of the section. Class variable of
        each subclass.
    """

    title: str = field(init=False)
    template: ClassVar[str]


@dataclass
class ErrorSection(Section):
    """
    A section that could not be built, shown with its error.

    Parameters
    ----------
    title : str
        Title of the section that could not be built.
    error : QAReportException
        Why the section could not be built.
    """

    template: ClassVar[str] = "sections/error_template.html"
    title: str = field(init=True)
    error: QAReportException


@dataclass
class MetadataSection(Section):
    """
    Metadata summary of a field.

    Parameters
    ----------
    file_information : FileInformation
        Measurement sets of the field and output folder of the run.
    observation_parameters : ObservationParameters
        Observation parameters of the field.
    station_configuration : StationConfiguration
        Stations of the field.
    """

    title: str = field(default="Metadata summary", init=False)
    template: ClassVar[str] = "sections/metadata.html"
    file_information: FileInformation
    observation_parameters: ObservationParameters
    station_configuration: StationConfiguration


@dataclass
class TableSection(Section):
    """
    A section with a sortable, paginated table.

    Parameters
    ----------
    title : str
        Header of the section.
    columns : list[Column | ColumnGroup]
        Columns of the table.
    rows : list[dict], optional
        Rows of the table.
    item_name : str
        Name of the rows, shown in the pagination, e.g. ``stations``.
    """

    template: ClassVar[str] = "sections/table.html"
    title: str = field(init=True)
    columns: list[Column] = field(default_factory=list)
    column_group: list[ColumnGroup] = field(default_factory=list)
    rows: list[dict] | None = None
    item_name: str = "stations"


@dataclass
class FlaggingStatisticsSection(TableSection):
    """
    Flagging statistics table, with the totals over all stations.

    Parameters
    ----------
    columns : list[Column | ColumnGroup]
        Columns of the table.
    rows : list[dict], optional
        Rows of the table.
    item_name : str
        Name of the rows, shown in the pagination, e.g. ``stations``.
    totals : dict[str, float | None], optional
        ``flagged_before`` and ``flagged_after`` average percentage.
    """

    title: str = field(default="Flagging Statistics", init=False)
    template: ClassVar[str] = "sections/flagging_statistics.html"
    totals: dict[str, float | None] | None = None


@dataclass
class CableDelaysSection(TableSection):
    """
    Cable delays table: the X and Y delay of each station.

    Parameters
    ----------
    columns : list[Column | ColumnGroup]
        Columns of the table.
    rows : list[dict], optional
        Rows of the table.
    item_name : str
        Name of the rows, shown in the pagination, e.g. ``stations``.
    """

    title: str = field(default="Cable Delays", init=False)
