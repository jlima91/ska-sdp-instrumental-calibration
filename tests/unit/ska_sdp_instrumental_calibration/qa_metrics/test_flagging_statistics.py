import numpy as np
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.qa_metrics.flagging_statistics import (
    build_flagging_rows,
    flagging_totals,
    gaintable_flagged_percentage,
)


@patch(
    "ska_sdp_instrumental_calibration.qa_metrics.flagging_statistics"
    ".read_h5parm_gains"
)
def test_should_get_flagged_percentage_per_station_from_gaintable(
    read_h5parm_gains_mock,
):
    weight = np.ones((1, 2, 2, 2, 2))
    weight[0, 1, 0, 0, 0] = 0
    soltab = MagicMock(name="soltab")
    soltab.antenna = ["station_a", "station_b"]
    read_h5parm_gains_mock.return_value = (None, weight, None, soltab)

    assert gaintable_flagged_percentage("gaintable.h5parm") == {
        "station_a": 0.0,
        "station_b": 25.0,
    }
    read_h5parm_gains_mock.assert_called_once_with("gaintable.h5parm")


def test_should_build_flagging_rows_per_station():
    rows = build_flagging_rows(
        {"0": "station_a", "1": "station_b"},
        {"station_a": 10.0, "station_b": 20.0},
        {"station_a": 15.0},
    )

    assert rows == [
        {
            "station_id": "0",
            "station_name": "station_a",
            "flagged_before": 10.0,
            "flagged_after": 15.0,
            "flagged_additional": 5.0,
        },
        {
            "station_id": "1",
            "station_name": "station_b",
            "flagged_before": 20.0,
            "flagged_after": None,
            "flagged_additional": None,
        },
    ]


def test_should_average_flagged_percentage_over_stations():
    rows = [
        {"flagged_before": 10.0, "flagged_after": None},
        {"flagged_before": 20.0, "flagged_after": None},
    ]

    assert flagging_totals(rows) == {
        "flagged_before": 15.0,
        "flagged_after": None,
    }
