import json

import numpy as np
import pytest

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.field_metadata import (  # noqa: E501
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.pre_flagging import (  # noqa: E501
    get_pre_flagging_paths,
    read_pre_flagging,
    write_pre_flagging,
)


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


def _field_metadata(sections):
    return FieldMetadata("field_a", "/qa/dir", sections=sections)


def test_should_get_pre_flagging_data_of_each_ms():
    field_metadata = _field_metadata(
        {
            "flagging": {
                "pre_flagging": {
                    "ms_a": "/qa/dir/derived/ms_a/pre-flagging-data.json",
                    "ms_b": "/qa/dir/derived/ms_b/pre-flagging-data.json",
                }
            }
        }
    )

    assert get_pre_flagging_paths(field_metadata) == [
        "/qa/dir/derived/ms_a/pre-flagging-data.json",
        "/qa/dir/derived/ms_b/pre-flagging-data.json",
    ]


def test_should_get_no_pre_flagging_data_if_none_is_recorded():
    # Not an error here: the reader raises for an empty list
    assert get_pre_flagging_paths(_field_metadata({})) == []


COUNTS = {
    "flagged": [1, 0, 1],
    "total": [2, 1, 1],
    "unsolvable": [0, 0, 1],
    "nchannels": 1,
}


def test_should_write_pre_flagging_counts_as_json(tmp_path):
    output_path = tmp_path / "pre-flagging-data.json"

    write_pre_flagging(COUNTS, str(output_path))

    assert json.loads(output_path.read_text()) == COUNTS


def test_should_read_back_written_pre_flagging(tmp_path):
    # The writer and reader share the file layout
    output_path = str(tmp_path / "pre-flagging-data.json")

    write_pre_flagging(COUNTS, output_path)
    (counts,) = read_pre_flagging([output_path], nstations=3)

    for key in ("flagged", "total", "unsolvable"):
        np.testing.assert_array_equal(counts[key], COUNTS[key])
    assert counts["nchannels"] == 1
