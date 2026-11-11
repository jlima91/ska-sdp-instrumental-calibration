import re

from astropy.time import Time


def mjd_seconds_to_utc(mjd_seconds: float) -> str:
    """
    Convert an MS time (MJD in seconds) to a UTC date and time string.

    Parameters
    ----------
    mjd_seconds : float
        Time in MJD seconds, as stored in the MS.

    Returns
    -------
    str
        UTC time, e.g. ``2026-02-12 10:00:00 UTC``.
    """
    utc = Time(mjd_seconds / 86400.0, format="mjd", scale="utc", precision=0)
    return f"{utc.iso} UTC"


def format_station_name(ms_station_name: str) -> str:
    """
    Get the smaller station name, e.g. S8-1, from a full station name.

    Parameters
    ----------
    ms_station_name : str
        Station name in the MS ANTENNA table.

    Returns
    -------
    str
        Familiar station name.
    """
    match = re.search(r"_([^_()]+)\)$", ms_station_name)
    return match.group(1) if match else ms_station_name


def format_metadata(metadata: dict) -> dict:
    """
    Format the QA metadata for display: start time in UTC and smaller
    station names.

    Parameters
    ----------
    metadata : dict
        QA metadata, as read from the JSON file.

    Returns
    -------
    dict
        Copy of the QA metadata with formatted values.
    """
    observation_parameters = {
        **metadata["observation_parameters"],
        "start_time": mjd_seconds_to_utc(
            metadata["observation_parameters"]["start_time"]
        ),
    }
    station_configuration = {
        station_id: format_station_name(ms_station_name)
        for station_id, ms_station_name in metadata[
            "station_configuration"
        ].items()
    }

    return {
        **metadata,
        "observation_parameters": observation_parameters,
        "station_configuration": station_configuration,
    }


def format_flagging_statistics(flagging_statistics: list[dict]) -> list[dict]:
    """
    Format the flagging statistics rows for display

    Parameters
    ----------
    flagging_statistics : list[dict]
        Rows of the flagging statistics table.

    Returns
    -------
    list[dict]
        Copy of the rows with familiar station names.
    """
    return [
        {**row, "station_name": format_station_name(row["station_name"])}
        for row in flagging_statistics
    ]
