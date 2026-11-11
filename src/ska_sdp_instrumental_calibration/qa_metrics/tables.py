from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Column:
    """
    A column of a QA metric table.

    Parameters
    ----------
    key : str
        Key of the column value in each table row.
    label : str
        Column header.
    unit : str, optional
        Unit of the values, shown in the header, e.g. ``%``.
    description : str, optional
        Description of the column, shown as tool tip of the header.
    value_format : {"text", "percentage"}
        How the values are displayed.
    """

    key: str
    label: str
    unit: str | None = None
    description: str | None = None
    value_format: Literal["text", "percentage"] = "text"


FLAGGING_STATISTICS_COLUMNS = [
    Column(
        "station_id",
        "Station ID",
        description="Index of the station in the measurement set",
    ),
    Column("station_name", "Station name"),
    Column(
        "flagged_before",
        "Flagged Before",
        unit="%",
        description="Gain solutions that cannot be solved, because all "
        "visibilities of the station in that channel are flagged in the "
        "measurement set",
        value_format="percentage",
    ),
    Column(
        "flagged_after",
        "Flagged After (final gaintable)",
        unit="%",
        description="Gain solutions flagged in the final gaintable",
        value_format="percentage",
    ),
    Column(
        "flagged_additional",
        "Flagged Additional",
        unit="%",
        description="Gain solutions flagged by the pipeline: "
        "Flagged After - Flagged Before",
        value_format="percentage",
    ),
]
