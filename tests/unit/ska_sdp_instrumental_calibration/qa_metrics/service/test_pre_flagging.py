import json

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.qa_metrics.domain.flagging_statistics import (  # noqa: E501
    PreFlagging,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.pre_flagging import (  # noqa: E501
    read_pre_flagging,
)
from ska_sdp_instrumental_calibration.qa_metrics.service.pre_flagging import (
    count_pre_flagging,
    export_pre_flagging,
    flagged_sample_counts,
    flagged_time_count,
    pre_flags_from_counts,
    unsolvable_gain_counts,
)


def _vis(flags, antenna1, antenna2, nstations):
    """
    Minimal visibility dataset: flags, the stations of each baseline and a
    configuration with nstations stations.
    """
    return xr.Dataset(
        {
            "flags": (
                ["time", "baselineid", "frequency", "polarisation"],
                da.from_array(flags, chunks=(1, *flags.shape[1:])),
            ),
        },
        coords={
            "antenna1": ("baselineid", antenna1),
            "antenna2": ("baselineid", antenna2),
        },
        # As in a Visibility, the configuration is kept in the attrs
        attrs={"configuration": xr.Dataset(coords={"id": range(nstations)})},
    )


def test_should_count_pre_flagging_per_station():
    # Baselines: 0-0 (auto), 0-1, 0-2, 1-2. 2 times, 2 channels, 2 pols
    flags = np.zeros((2, 4, 2, 2), dtype=bool)
    # Auto-correlation flags are ignored
    flags[:, 0] = True
    # 0-1 flagged at time 0, channel 0, in one polarisation only
    flags[0, 1, 0, 1] = True
    # 0-2 and 1-2 flagged at all times in channel 1: station 2 has no data
    # left in channel 1, so its gain there cannot be solved
    flags[:, 2:, 1] = True
    vis = _vis(flags, [0, 0, 0, 1], [0, 1, 2, 2], nstations=3)

    counts = count_pre_flagging(vis)

    # Each station has 2 cross baselines of 2 times x 2 channels: 8 samples
    assert counts == {
        "flagged": [3, 3, 4],
        "total": [8, 8, 8],
        "unsolvable": [0, 0, 1],
        "nchannels": 2,
    }
    # Plain ints, so the counts can be written as JSON
    assert json.loads(json.dumps(counts)) == counts


def test_should_export_counts_readable_by_the_repository(tmp_path):
    flags = np.zeros((1, 2, 1, 1), dtype=bool)
    flags[0, 1] = True
    vis = _vis(flags, [0, 0], [1, 2], nstations=3)
    output_path = str(tmp_path / "pre-flagging-data.json")

    export_pre_flagging(vis, output_path)
    (counts,) = read_pre_flagging([output_path], nstations=3)

    np.testing.assert_array_equal(counts["flagged"], [1, 0, 1])
    np.testing.assert_array_equal(counts["total"], [2, 1, 1])


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


STATION_NAMES = ["station_a", "station_b"]


def _counts(flagged, total, unsolvable, nchannels):
    """Counts of one MS, as read by read_pre_flagging"""
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

    pre_flagging = pre_flags_from_counts(counts, STATION_NAMES)

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

    pre_flagging = pre_flags_from_counts(counts, STATION_NAMES)

    assert pre_flagging.flagged_data["station_a"] == pytest.approx(40.0)
    assert pre_flagging.unsolvable_gains["station_a"] == pytest.approx(25.0)
