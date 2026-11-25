from dataclasses import dataclass
from typing import Literal

from ..domain.qa_metadata import StationConfiguration


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
    value_format : {"text", "percentage", "number"}
        How the values are displayed.
    threshold : float, optional
        Acceptable threshold of the values. Values below it fail QA and are
        shown in red.
    """

    key: str
    label: str
    unit: str | None = None
    description: str | None = None
    value_format: Literal["text", "percentage", "number"] = "text"
    threshold: float | None = None


@dataclass(frozen=True)
class ColumnGroup:
    """
    Columns of a QA metric table shown under a shared header, e.g. the X
    and Y polarisation of a metric.

    Parameters
    ----------
    label : str
        Shared header, shown above the headers of the columns.
    columns : list[Column]
        Columns of the group.
    description : str, optional
        Description of the group, shown as tool tip of the shared header.
    """

    label: str
    columns: list[Column]
    description: str | None = None


def build_station_rows(
    station_configuration: StationConfiguration,
    values_per_station: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Rows of a QA metric table, one per station, with the smaller station
    names for display.

    Parameters
    ----------
    station_configuration : StationConfiguration
        Stations of the field.
    values_per_station : dict[str, dict], optional
        Values of the row keyed by column key, per MS station name. A
        station without values gets a row with only its ID and name.

    Returns
    -------
    list[dict]
        ``station_id``, ``station_name`` and the station values per station.
    """
    values_per_station = values_per_station or {}
    return [
        {
            "station": value,
            # Values are keyed by the MS station name
            **values_per_station.get(key, {}),
        }
        for key, value in (station_configuration.station_names.items())
    ]


def _polarisation_columns(
    key: str, unit: str | None = None, threshold: float | None = None
) -> list[Column]:
    """X and Y polarisation columns of a metric, keyed x_<key> and y_<key>"""
    return [
        Column(
            f"{pol}_{key}",
            pol.upper(),
            unit=unit,
            value_format="number",
            threshold=threshold,
        )
        for pol in ("x", "y")
    ]


STATION_COLUMN = Column("station", "Station ID")
FLAGGING_STATISTICS_COLUMNS = [
    STATION_COLUMN,
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


def get_bandpass_fit_metrics_columns(
    phase_fit_threshold: float, gain_fit_threshold: float
) -> list[Column | ColumnGroup]:
    return [
        STATION_COLUMN,
        ColumnGroup(
            "Phase Fit Quality",
            _polarisation_columns(
                "phase_fit_quality", threshold=phase_fit_threshold
            ),
            description="Value between 0-1 representing quality of phase "
            "fit in X/Y polarisation; ratio of number of channels with "
            "distance to fit smaller than 5 standard deviations to number of "
            "fitted channels.\n"
            "The phase is fitted with a joint linear delay + 1/ν TEC model "
            "across the observing band.\n"
            "A robust estimator of the standard deviation should be used to "
            "ensure that the metric is sensitive to outliers. The standard "
            "deviation could be estimated from the interquartile range "
            "(IQR), i.e. σ = IQR/1.349. The optimal σ level used to identify "
            "bad channels (initially set to 5) will need to be determined for "
            "SKA-Low data.",
        ),
        ColumnGroup(
            "Gain Fit Quality",
            _polarisation_columns(
                "gains_fit_quality", threshold=gain_fit_threshold
            ),
            description="Value between 0-1 representing quality of gain "
            "amplitude fit in X/Y polarisation; ratio of number of channels "
            "with distance to fit smaller than 5 standard deviations to "
            "number of fitted channels.\n"
            "The gain amplitude is fitted with a 1st order polynomial per "
            "SKA-Low coarse channel (0.78 MHz). Wider band fitting may be "
            "appropriate for coarse channels affected by RFI.\n"
            "A robust estimator of the standard deviation should be used to "
            "ensure that the metric is sensitive to outliers. The standard "
            "deviation could be estimated from the IQR, i.e. σ = IQR/1.349. "
            "The optimal σ level used to identify bad channels (initially set "
            "to 5) will need to be determined for SKA-Low data.",
        ),
    ]


CABLE_DELAYS_COLUMNS = [
    STATION_COLUMN,
    Column("x_delay", "X Delay", unit="ns", value_format="number"),
    Column("y_delay", "Y Delay", unit="ns", value_format="number"),
]
