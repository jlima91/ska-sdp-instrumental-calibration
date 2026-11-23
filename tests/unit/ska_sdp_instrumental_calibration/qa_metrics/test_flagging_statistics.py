import numpy as np

from ska_sdp_instrumental_calibration.qa_metrics.domain.qa_metric import (
    FlaggingStatistics,
)


def test_should_get_flagged_percentage_after_from_parallel_hand_weights():
    # (time, antenna, frequency, 2, 2): 2 times, 2 stations, 2 channels
    weight = np.ones((2, 2, 2, 2, 2))
    # station_a: 1 of its 8 parallel hand gains is flagged (XX, t0, c0)
    weight[0, 0, 0, 0, 0] = 0
    # Cross hands (XY, YX) are not counted
    weight[:, 1, :, 0, 1] = 0
    weight[:, 1, :, 1, 0] = 0
    flagged_before = {"station_a": 0.0, "station_b": 10.0}

    statistics = FlaggingStatistics.from_weights(
        weight, ["station_a", "station_b"], flagged_before
    )

    assert statistics == FlaggingStatistics(
        flagged_before=flagged_before,
        flagged_after={"station_a": 12.5, "station_b": 0.0},
    )
