import numpy as np
import pytest

from ska_sdp_instrumental_calibration.qa_metrics.domain import PreFlagging

STATION_NAMES = ["station_a", "station_b"]


def _counts(flagged, total, unsolvable, nchannels):
    """Counts of one MS, as read by products.read_pre_flagging"""
    return {
        "flagged": np.asarray(flagged),
        "total": np.asarray(total),
        "unsolvable": np.asarray(unsolvable),
        "nchannels": nchannels,
    }


def test_should_get_pre_flagging_percentages_per_station():
    counts = [
        _counts(flagged=[2, 0], total=[8, 8], unsolvable=[1, 0], nchannels=4)
    ]

    pre_flagging = PreFlagging.from_counts(counts, STATION_NAMES)

    assert pre_flagging == PreFlagging(
        flagged_data={"station_a": 25.0, "station_b": 0.0},
        unsolvable_gains={"station_a": 25.0, "station_b": 0.0},
    )


def test_should_sum_counts_of_all_ms_before_dividing():
    # MS a is 4 times bigger than MS b: averaging the percentages (50% and
    # 0%) would give 25%, summing the counts gives 4 / 10 = 40%
    counts = [
        _counts(flagged=[4, 0], total=[8, 8], unsolvable=[1, 0], nchannels=2),
        _counts(flagged=[0, 0], total=[2, 2], unsolvable=[0, 0], nchannels=2),
    ]

    pre_flagging = PreFlagging.from_counts(counts, STATION_NAMES)

    assert pre_flagging.flagged_data["station_a"] == pytest.approx(40.0)
    assert pre_flagging.unsolvable_gains["station_a"] == pytest.approx(25.0)
