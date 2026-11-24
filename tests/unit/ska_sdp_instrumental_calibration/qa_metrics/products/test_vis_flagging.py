import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.qa_metrics.products.vis_flagging import (
    flagged_sample_counts,
    flagged_time_count,
    unsolvable_gain_counts,
)


def test_should_count_flagged_time_steps_per_baseline_and_channel():
    flags = np.zeros((2, 1, 2, 2), dtype=bool)
    flags[0, 0, 0, 1] = True
    flags[:, 0, 1, 0] = True
    vis = xr.Dataset(
        {
            "flags": (
                ["time", "baselineid", "frequency", "polarisation"],
                flags,
            )
        }
    )

    np.testing.assert_array_equal(flagged_time_count(vis), [[1, 2]])


def test_should_count_flagged_samples_per_station():
    flagged, total = flagged_sample_counts(
        np.array([[2, 0], [0, 0]]), np.array([0, 0]), np.array([1, 2]), 2, 3
    )

    np.testing.assert_array_equal(flagged, [2, 2, 0])
    np.testing.assert_array_equal(total, [8, 4, 4])


def test_should_count_unsolvable_gains_per_station():
    unsolvable, nchannels = unsolvable_gain_counts(
        np.array([[2, 0], [0, 0]]), np.array([0, 0]), np.array([1, 2]), 2, 3
    )

    np.testing.assert_array_equal(unsolvable, [0, 1, 0])
    assert nchannels == 2
