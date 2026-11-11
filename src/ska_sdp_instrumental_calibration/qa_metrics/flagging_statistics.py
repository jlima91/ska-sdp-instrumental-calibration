from ..data_managers.h5parm import read_h5parm_gains


def gaintable_flagged_percentage(h5parm_path: str) -> dict[str, float]:
    """
    Percentage of flagged gain solutions per station in an h5parm.

    A gain is flagged where its weight is 0. Only the parallel hand gains
    (XX, YY) solved by bandpass are counted.

    Parameters
    ----------
    h5parm_path : str
        Path to the h5parm file.

    Returns
    -------
    dict[str, float]
        Flagged percentage keyed by station name, as in the MS.
    """
    _, weight, _, soltab = read_h5parm_gains(h5parm_path)

    # weight axes are (time, station, frequency, receptor, receptor)
    is_flagged = weight[..., [0, 1], [0, 1]] == 0
    # per station: sum over time, frequency and receptor
    flagged = is_flagged.sum(axis=(0, 2, 3))
    total = is_flagged[:, 0, ...].size

    return dict(zip(soltab.antenna, flagged / total * 100))


def build_flagging_rows(
    station_configuration: dict[str, str],
    flagged_before: dict[str, float],
    flagged_after: dict[str, float],
) -> list[dict]:
    """
    Rows of the flagging statistics table, one per station.

    Parameters
    ----------
    station_configuration : dict[str, str]
        Station names, as in the MS, keyed by station ID.
    flagged_before : dict[str, float]
        Percentage of gains that cannot be solved because all their
        visibilities are flagged, keyed by station name.
    flagged_after : dict[str, float]
        Flagged percentage of the gains in the final gaintable, keyed by
        station name.

    Returns
    -------
    list[dict]
        ``station_id``, ``station_name``, ``flagged_before``,
        ``flagged_after`` and ``flagged_additional`` per station.
        A percentage is None where it is not available.
    """
    rows = []
    for station_id, station_name in station_configuration.items():
        before = flagged_before.get(station_name)
        after = flagged_after.get(station_name)
        additional = (
            after - before
            if before is not None and after is not None
            else None
        )
        rows.append(
            {
                "station_id": station_id,
                "station_name": station_name,
                "flagged_before": before,
                "flagged_after": after,
                "flagged_additional": additional,
            }
        )

    return rows
