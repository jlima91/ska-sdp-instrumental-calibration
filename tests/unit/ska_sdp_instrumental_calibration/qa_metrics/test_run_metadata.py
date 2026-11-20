import json
from pathlib import Path

import pytest
from mock import patch

from ska_sdp_instrumental_calibration.exceptions import QAMetadataError
from ska_sdp_instrumental_calibration.qa_metrics.qa_section import QASection
from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
    RunMetadata,
)


def test_should_return_none_for_label_without_data():
    field_metadata = FieldMetadata("field", "qa_dir")

    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) is None
    assert field_metadata.get_data_for(QASection.BANDPASS.GAINTABLE) is None


def test_should_delegate_storing_and_loading_to_the_label():
    field_metadata = FieldMetadata("field", "qa_dir")
    label = QASection.BANDPASS.GAINTABLE

    with patch.object(
        type(label), "store", autospec=True
    ) as store_mock, patch.object(
        type(label), "load", autospec=True, return_value={"a": ["a.h5"]}
    ) as load_mock:
        field_metadata.add_data_for(label, "a.h5", prefix="a")
        data = field_metadata.get_data_for(label)

    store_mock.assert_called_once_with(
        label, field_metadata.sections["bandpass"], "a.h5", prefix="a"
    )
    load_mock.assert_called_once_with(
        label, field_metadata.sections["bandpass"]
    )
    assert data == {"a": ["a.h5"]}


def test_should_store_data_in_the_shape_of_each_label_level():
    field_metadata = FieldMetadata("field", "qa_dir")

    field_metadata.add_data_for(QASection.METADATA_SUMMARY.CHANNELS, 32)
    field_metadata.add_data_for(QASection.METADATA_SUMMARY.CHANNELS, 64)
    field_metadata.add_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")
    field_metadata.add_data_for(QASection.METADATA_SUMMARY.MS, "b.ms")
    field_metadata.add_data_for(
        QASection.BANDPASS.GAINTABLE, "a.h5", prefix="a"
    )
    field_metadata.add_data_for(
        QASection.BANDPASS.GAINTABLE, "a_1.h5", prefix="a"
    )

    assert field_metadata.sections == {
        "metadata_summary": {"channels": 64, "MS": ["a.ms", "b.ms"]},
        "bandpass": {"gaintable": {"a": ["a.h5", "a_1.h5"]}},
    }


def test_should_raise_label_errors_through_add_data_for():
    field_metadata = FieldMetadata("field", "qa_dir")

    with pytest.raises(QAMetadataError, match="needs a prefix"):
        field_metadata.add_data_for(QASection.BANDPASS.GAINTABLE, "a.h5")

    with pytest.raises(QAMetadataError, match="expects str"):
        # A Path is not JSON serialisable: passed on purpose to check the
        # label rejects it at runtime
        field_metadata.add_data_for(
            QASection.METADATA_SUMMARY.MS,
            Path("a.ms"),  # pyright: ignore[reportArgumentType]
        )


def test_should_store_path_like_qa_dir_as_str():
    field_metadata = FieldMetadata("field", Path("qa_dir"))

    assert field_metadata.qa_dir == "qa_dir"


def test_should_convert_field_metadata_to_dict():
    field_metadata = FieldMetadata("field", "qa_dir")
    field_metadata.add_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")

    assert field_metadata.to_dict() == {
        "qa_dir": "qa_dir",
        "field_name": "field",
        "sections": {"metadata_summary": {"MS": ["a.ms"]}},
    }


def test_should_create_field_metadata_from_its_dict():
    field_metadata = FieldMetadata("field", "qa_dir")
    field_metadata.add_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")

    assert (
        FieldMetadata(**field_metadata.to_dict()).to_dict()
        == field_metadata.to_dict()
    )


def test_should_record_data_per_field():
    run_metadata = RunMetadata("qa_dir")

    run_metadata.add_for_field(
        "field_a", QASection.METADATA_SUMMARY.MS, "a.ms"
    )
    run_metadata.add_for_field(
        "field_b", QASection.METADATA_SUMMARY.MS, "b.ms"
    )
    run_metadata.add_for_field(
        "field_a", QASection.BANDPASS.GAINTABLE, "a.h5", prefix="a"
    )

    field_a = run_metadata.field_metadata["field_a"]
    field_b = run_metadata.field_metadata["field_b"]
    assert field_a.field_name == "field_a"
    assert field_a.qa_dir == "qa_dir"
    assert field_a.get_data_for(QASection.METADATA_SUMMARY.MS) == ["a.ms"]
    assert field_a.get_data_for(QASection.BANDPASS.GAINTABLE) == {
        "a": ["a.h5"]
    }
    assert field_b.get_data_for(QASection.METADATA_SUMMARY.MS) == ["b.ms"]
    assert [metadata.field_name for metadata in run_metadata] == [
        "field_a",
        "field_b",
    ]


def test_should_write_run_metadata_to_json(tmp_path):
    run_metadata = RunMetadata(str(tmp_path))
    run_metadata.add_for_field("field", QASection.METADATA_SUMMARY.MS, "a.ms")

    run_metadata.to_json()

    with open(tmp_path / "run_metadata.json", encoding="utf-8") as f:
        assert json.load(f) == {
            "field": {
                "qa_dir": str(tmp_path),
                "field_name": "field",
                "sections": {"metadata_summary": {"MS": ["a.ms"]}},
            }
        }


def test_should_read_run_metadata_written_to_json(tmp_path):
    run_metadata = RunMetadata(tmp_path)
    run_metadata.add_for_field(
        "field_a", QASection.METADATA_SUMMARY.CHANNELS, 64
    )
    run_metadata.add_for_field(
        "field_a", QASection.METADATA_SUMMARY.MS, "a.ms"
    )
    run_metadata.add_for_field(
        "field_b", QASection.BANDPASS.GAINTABLE, "b.h5", prefix="b"
    )
    run_metadata.to_json()

    read_metadata = RunMetadata.initialize(tmp_path)

    assert read_metadata.qa_dir == tmp_path
    assert {
        name: metadata.to_dict()
        for name, metadata in read_metadata.field_metadata.items()
    } == {
        name: metadata.to_dict()
        for name, metadata in run_metadata.field_metadata.items()
    }
    field_a = read_metadata.field_metadata["field_a"]
    field_b = read_metadata.field_metadata["field_b"]
    assert field_a.get_data_for(QASection.METADATA_SUMMARY.CHANNELS) == 64
    assert field_a.get_data_for(QASection.METADATA_SUMMARY.MS) == ["a.ms"]
    assert field_b.get_data_for(QASection.BANDPASS.GAINTABLE) == {
        "b": ["b.h5"]
    }


def test_should_raise_if_no_run_metadata_to_read(tmp_path):
    with pytest.raises(QAMetadataError, match="No run metadata"):
        RunMetadata.initialize(tmp_path)
