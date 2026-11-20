import json

import pytest
from mock import patch

from ska_sdp_instrumental_calibration.qa_metrics.qa_metric import (
    generate_qa_metric,
)


@pytest.fixture
def qa_metadata(tmp_path):
    metadata = {
        "source_name": "source_name",
        "input_ms": ["/path/to/vis.ms"],
        "output_path": str(tmp_path),
        "observation_parameters": {
            "start_time": 5e9,
            "observation_length": 20.0,
            "integration_time": 1.0,
            "channels": 2,
        },
        "pre_flagged_data": {"station_a": 10.0},
        "pre_flagged_gains": {"station_a": 5.0},
        "station_configuration": {"0": "station_a"},
    }
    (tmp_path / "source_name_qa-metadata-data.json").write_text(
        json.dumps(metadata)
    )


@pytest.mark.skip(reason="Needs update")
@patch(
    "ska_sdp_instrumental_calibration.qa_metrics.qa_metric"
    ".gaintable_flagged_percentage"
)
def test_should_generate_qa_metric_with_flagging_statistics(
    gaintable_flagged_percentage_mock, tmp_path, qa_metadata
):
    (tmp_path / "source_name_gaintable.h5parm").touch()
    gaintable_flagged_percentage_mock.return_value = {"station_a": 7.5}

    generate_qa_metric(str(tmp_path))

    page = (tmp_path / "inst-source_name-qa-metric.html").read_text()
    gaintable_flagged_percentage_mock.assert_called_once_with(
        str(tmp_path / "source_name_gaintable.h5parm")
    )
    assert "7.50%" in page
