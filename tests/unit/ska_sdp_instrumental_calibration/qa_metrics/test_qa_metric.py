import json

import pytest
from mock import patch

from ska_sdp_instrumental_calibration.qa_metrics.qa_metric import (
    GAINTABLE_MISSING,
    PAGE_TITLE,
    PREREQUISITE_DATA_MISSING,
    SECTION_TITLES,
    generate_qa_metric,
    write_qa_metric,
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


def test_should_write_qa_metric_page_named_after_source_name(tmp_path):
    write_qa_metric("<html></html>", str(tmp_path), "source_name")

    assert (
        tmp_path / "inst-source_name-qa-metric.html"
    ).read_text() == "<html></html>"


def test_should_generate_qa_metric_with_errors_when_no_qa_metadata(tmp_path):
    generate_qa_metric(str(tmp_path))

    page = (
        tmp_path / "inst-PLACEHOLDER_SOURCE_NAME-qa-metric.html"
    ).read_text()
    assert PAGE_TITLE in page
    assert all(title in page for title in SECTION_TITLES)
    assert PREREQUISITE_DATA_MISSING in page


def test_should_generate_qa_metric_with_error_when_gaintable_missing(
    tmp_path, qa_metadata
):
    generate_qa_metric(str(tmp_path))

    page = (tmp_path / "inst-source_name-qa-metric.html").read_text()
    assert "station_a" in page
    assert GAINTABLE_MISSING in page


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
    assert GAINTABLE_MISSING not in page
