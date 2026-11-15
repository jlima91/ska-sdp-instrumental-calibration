import numpy as np
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.data_export.qa_metadata import (  # noqa: E501
    get_ms_qa_data,
    get_qa_metadata_paths,
    read_qa_metadata,
    write_qa_metadata,
)


@patch(
    "ska_sdp_instrumental_calibration.data_managers.data_export.qa_metadata"
    ".flagged_time_count"
)
def test_should_get_ms_qa_data(flagged_time_count_mock):
    vis = MagicMock(name="vis")
    vis.sizes = {"time": 2, "frequency": 4}
    vis.attrs = {"configuration": MagicMock(name="configuration")}

    ms_qa_data = get_ms_qa_data(vis, "/path/to/vis.ms")

    flagged_time_count_mock.assert_called_once_with(vis)
    assert ms_qa_data == {
        "input_ms": "/path/to/vis.ms",
        "timestamps": vis.time.data,
        "channels": 4,
        "station_names": vis.attrs["configuration"].names.data,
        "antenna1": vis.antenna1.data,
        "antenna2": vis.antenna2.data,
        "flagged_time_count": flagged_time_count_mock.return_value.data,
        "ntimes": 2,
    }


def test_should_write_and_read_qa_metadata(tmp_path):
    ms_qa_data = {
        "input_ms": "/path/to/vis.ms",
        "timestamps": np.array([10.0, 12.0]),
        "channels": 2,
        "station_names": np.array(["a", "b"]),
        "antenna1": np.array([0]),
        "antenna2": np.array([1]),
        "flagged_time_count": np.array([[2, 0]]),
        "ntimes": 2,
    }

    write_qa_metadata("field", [ms_qa_data], str(tmp_path))

    qa_metadata_paths = get_qa_metadata_paths(str(tmp_path))
    assert qa_metadata_paths == [str(tmp_path / "field_qa-metadata-data.json")]
    assert read_qa_metadata(qa_metadata_paths[0]) == {
        "source_name": "field",
        "input_ms": ["/path/to/vis.ms"],
        "output_path": str(tmp_path),
        "observation_parameters": {
            "start_time": 9.0,
            "observation_length": 4.0,
            "integration_time": 2.0,
            "channels": 2,
        },
        "pre_flagged_data": {"a": 50.0, "b": 50.0},
        "pre_flagged_gains": {"a": 50.0, "b": 50.0},
        "station_configuration": {"0": "a", "1": "b"},
    }
