import json
from pathlib import Path

from ska_sdp_instrumental_calibration.qa_metrics.qa_section import QASection
from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
    RunMetadata,
)


def test_should_return_none_for_label_without_data():
    field_metadata = FieldMetadata("field", "qa_dir")

    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) is None


def test_should_replace_data_of_label_on_add():
    field_metadata = FieldMetadata("field", "qa_dir")

    field_metadata.add_data_for(QASection.METADATA_SUMMARY.MS, ["a.ms"])
    field_metadata.add_data_for(
        QASection.METADATA_SUMMARY.MS, ["b.ms", "c.ms"]
    )

    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) == [
        "b.ms",
        "c.ms",
    ]


def test_should_keep_data_of_label_on_append():
    field_metadata = FieldMetadata("field", "qa_dir")

    field_metadata.append_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")
    field_metadata.append_data_for(QASection.METADATA_SUMMARY.MS, "b.ms")

    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) == [
        "a.ms",
        "b.ms",
    ]


def test_should_keep_labels_of_a_section_separate():
    field_metadata = FieldMetadata("field", "qa_dir")

    field_metadata.append_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")
    field_metadata.append_data_for(
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA, "a.json"
    )

    assert field_metadata.sections == {
        "metadata_summary": {"MS": ["a.ms"], "visibility_metadata": ["a.json"]}
    }


def test_should_store_path_like_data_as_str():
    field_metadata = FieldMetadata("field", "qa_dir")

    field_metadata.add_data_for(QASection.METADATA_SUMMARY.MS, [Path("a.ms")])
    field_metadata.append_data_for(
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA, Path("a.json")
    )

    assert field_metadata.get_data_for(QASection.METADATA_SUMMARY.MS) == [
        "a.ms"
    ]
    assert field_metadata.get_data_for(
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA
    ) == ["a.json"]


def test_should_store_path_like_qa_dir_as_str():
    field_metadata = FieldMetadata("field", Path("qa_dir"))

    assert field_metadata.qa_dir == "qa_dir"


def test_should_convert_field_metadata_to_dict():
    field_metadata = FieldMetadata("field", "qa_dir")
    field_metadata.append_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")

    assert field_metadata.to_dict() == {
        "qa_dir": "qa_dir",
        "field_name": "field",
        "sections": {"metadata_summary": {"MS": ["a.ms"]}},
    }


def test_should_create_field_metadata_from_its_dict():
    field_metadata = FieldMetadata("field", "qa_dir")
    field_metadata.append_data_for(QASection.METADATA_SUMMARY.MS, "a.ms")

    assert (
        FieldMetadata(**field_metadata.to_dict()).to_dict()
        == field_metadata.to_dict()
    )


def test_should_record_data_per_field():
    run_metadata = RunMetadata("qa_dir")

    run_metadata.append_for_field(
        "field_a", QASection.METADATA_SUMMARY.MS, "a.ms"
    )
    run_metadata.append_for_field(
        "field_b", QASection.METADATA_SUMMARY.MS, "b.ms"
    )
    run_metadata.add_for_field(
        "field_a", QASection.METADATA_SUMMARY.MS, ["c.ms"]
    )

    field_a = run_metadata.field_metadata["field_a"]
    field_b = run_metadata.field_metadata["field_b"]
    assert field_a.field_name == "field_a"
    assert field_a.qa_dir == "qa_dir"
    assert field_a.get_data_for(QASection.METADATA_SUMMARY.MS) == ["c.ms"]
    assert field_b.get_data_for(QASection.METADATA_SUMMARY.MS) == ["b.ms"]


def test_should_append_data_of_several_ms_of_same_field():
    run_metadata = RunMetadata("qa_dir")

    run_metadata.append_for_field(
        "field", QASection.METADATA_SUMMARY.MS, "a.ms"
    )
    run_metadata.append_for_field(
        "field", QASection.METADATA_SUMMARY.MS, "b.ms"
    )

    assert run_metadata.field_metadata["field"].get_data_for(
        QASection.METADATA_SUMMARY.MS
    ) == ["a.ms", "b.ms"]


def test_should_write_run_metadata_to_json(tmp_path):
    run_metadata = RunMetadata(str(tmp_path))
    run_metadata.append_for_field(
        "field", QASection.METADATA_SUMMARY.MS, "a.ms"
    )

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
    run_metadata.append_for_field(
        "field_a", QASection.METADATA_SUMMARY.MS, Path("a.ms")
    )
    run_metadata.append_for_field(
        "field_b", QASection.METADATA_SUMMARY.VISIBILITY_METADATA, "b.json"
    )
    run_metadata.to_json()

    read_metadata = RunMetadata.from_json(tmp_path)

    assert read_metadata.qa_dir == tmp_path
    assert {
        name: metadata.to_dict()
        for name, metadata in read_metadata.field_metadata.items()
    } == {
        name: metadata.to_dict()
        for name, metadata in run_metadata.field_metadata.items()
    }
    assert read_metadata.field_metadata["field_a"].get_data_for(
        QASection.METADATA_SUMMARY.MS
    ) == ["a.ms"]
