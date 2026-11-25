import json

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.service import (
    export_pre_flagging,
    read_pre_flagging,
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


def test_should_export_pre_flagging_counts_per_station(tmp_path):
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
    output_path = tmp_path / "pre-flagging-data.json"

    export_pre_flagging(vis, str(output_path))

    # Each station has 2 cross baselines of 2 times x 2 channels: 8 samples
    assert json.loads(output_path.read_text()) == {
        "flagged": [3, 3, 4],
        "total": [8, 8, 8],
        "unsolvable": [0, 0, 1],
        "nchannels": 2,
    }


def _write_counts(path, flagged, total, unsolvable, nchannels):
    path.write_text(
        json.dumps(
            {
                "flagged": flagged,
                "total": total,
                "unsolvable": unsolvable,
                "nchannels": nchannels,
            }
        )
    )
    return str(path)


def test_should_read_pre_flagging_counts_of_each_ms(tmp_path):
    paths = [
        _write_counts(
            tmp_path / "a.json",
            flagged=[2, 0],
            total=[8, 8],
            unsolvable=[1, 0],
            nchannels=4,
        ),
        _write_counts(
            tmp_path / "b.json",
            flagged=[0, 1],
            total=[2, 2],
            unsolvable=[0, 0],
            nchannels=4,
        ),
    ]

    counts = read_pre_flagging(paths, nstations=2)

    assert len(counts) == 2
    np.testing.assert_array_equal(counts[0]["flagged"], [2, 0])
    np.testing.assert_array_equal(counts[1]["flagged"], [0, 1])
    assert counts[0]["nchannels"] == 4


def test_should_read_back_exported_pre_flagging(tmp_path):
    flags = np.zeros((1, 2, 1, 1), dtype=bool)
    flags[0, 1] = True
    vis = _vis(flags, [0, 0], [1, 2], nstations=3)
    output_path = str(tmp_path / "pre-flagging-data.json")

    export_pre_flagging(vis, output_path)
    (counts,) = read_pre_flagging([output_path], nstations=3)

    np.testing.assert_array_equal(counts["flagged"], [1, 0, 1])
    np.testing.assert_array_equal(counts["total"], [2, 1, 1])


def test_should_raise_if_no_pre_flagging_files():
    with pytest.raises(RunMetadataError, match="No pre-flagging data"):
        read_pre_flagging([], nstations=2)


def test_should_raise_if_pre_flagging_file_was_not_produced(tmp_path):
    with pytest.raises(RunMetadataError, match="was not produced"):
        read_pre_flagging([str(tmp_path / "missing.json")], nstations=2)


def test_should_raise_if_counts_are_not_one_per_station(tmp_path):
    path = _write_counts(
        tmp_path / "a.json",
        flagged=[0],
        total=[8],
        unsolvable=[0],
        nchannels=4,
    )

    with pytest.raises(RunMetadataError, match="1 flagged counts for 2"):
        read_pre_flagging([path], nstations=2)


def test_should_raise_if_pre_flagging_file_cannot_be_read(tmp_path):
    # A file left half written by a failed export
    path = tmp_path / "a.json"
    path.write_text('{"flagged": [1')

    with pytest.raises(RunMetadataError, match="cannot be read"):
        read_pre_flagging([str(path)], nstations=1)


def test_should_raise_if_pre_flagging_counts_are_missing(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"flagged": [1], "total": [2]}))

    with pytest.raises(
        RunMetadataError, match="has no unsolvable, nchannels counts"
    ):
        read_pre_flagging([str(path)], nstations=1)
