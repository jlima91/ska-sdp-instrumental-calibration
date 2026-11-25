import numpy as np
import pytest

from ska_sdp_instrumental_calibration.qa_metrics.domain.qa_metric import (
    CableDelays,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable


def _delaytable(delay):
    delay = np.asarray(delay, dtype=float)
    return DelayTable.constructor(
        delay=delay,
        offset=np.zeros_like(delay),
        time=np.arange(delay.shape[0], dtype=float),
        antenna_name=np.array(["station_a", "station_b"]),
        pol=["XX", "YY"],
    )


def test_should_take_median_delay_over_time_in_nanoseconds():
    # (time, antenna, pol): 3 times, 2 stations, XX and YY in seconds
    delaytable = _delaytable(
        [
            [[10e-9, 30e-9], [1e-9, 2e-9]],
            [[20e-9, 40e-9], [3e-9, 4e-9]],
            [[30e-9, 50e-9], [5e-9, 6e-9]],
        ]
    )

    cable_delays = CableDelays.from_delaytable(delaytable)

    assert cable_delays.values_per_station == {
        "station_a": {
            "x_delay": pytest.approx(20.0),
            "y_delay": pytest.approx(40.0),
        },
        "station_b": {
            "x_delay": pytest.approx(3.0),
            "y_delay": pytest.approx(4.0),
        },
    }


def test_should_not_be_pulled_by_one_badly_solved_scan():
    # station_a: one scan is off by 90 ns; the mean would be 40 ns
    delaytable = _delaytable(
        [
            [[10e-9, 10e-9], [1e-9, 1e-9]],
            [[10e-9, 10e-9], [1e-9, 1e-9]],
            [[100e-9, 10e-9], [1e-9, 1e-9]],
        ]
    )

    values = CableDelays.from_delaytable(delaytable).values_per_station

    assert values["station_a"]["x_delay"] == pytest.approx(10.0)


def test_should_leave_unsolved_delays_out_of_the_median():
    delaytable = _delaytable(
        [
            [[10e-9, np.nan], [1e-9, 1e-9]],
            [[np.nan, 40e-9], [1e-9, 1e-9]],
            [[30e-9, 60e-9], [1e-9, 1e-9]],
        ]
    )

    values = CableDelays.from_delaytable(delaytable).values_per_station

    # Medians of (10, 30) and (40, 60), the NaN left out
    assert values["station_a"]["x_delay"] == pytest.approx(20.0)
    assert values["station_a"]["y_delay"] == pytest.approx(50.0)
