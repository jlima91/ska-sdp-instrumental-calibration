from typing import TypedDict

import pytest

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.qa_section import (
    LabelSpec,
    Level,
    QASection,
    QASectionLabel,
)


def test_should_derive_section_key_from_class_name():
    assert QASection.METADATA_SUMMARY.MS.section == "metadata_summary"
    assert QASection.METADATA_SUMMARY.TIMESTAMPS.section == "metadata_summary"
    assert (
        QASection.METADATA_SUMMARY.FINAL_GAINTABLE.section
        == "metadata_summary"
    )
    assert QASection.FLAGGING.GAIN_FITS.section == "flagging"
    assert QASection.FLAGGING.GAINTABLE.section == "flagging"
    assert QASection.BANDPASS.GAINTABLE.section == "bandpass"
    assert QASection.DELAY.GAINTABLE.section == "delay"
    assert QASection.DELAY.DELAYTABLE.section == "delay"


def test_should_use_label_value_as_plain_string():
    assert QASection.METADATA_SUMMARY.MS == "MS"
    assert QASection.METADATA_SUMMARY.TIMESTAMPS == "timestamps"
    assert QASection.METADATA_SUMMARY.FINAL_GAINTABLE == "final_gaintable"
    assert QASection.FLAGGING.GAIN_FITS == "gain_fits"
    assert QASection.FLAGGING.GAINTABLE == "gaintable"
    assert QASection.BANDPASS.GAINTABLE == "gaintable"
    assert QASection.DELAY.GAINTABLE == "gaintable"
    assert QASection.DELAY.DELAYTABLE == "delaytable"


def test_should_keep_same_label_of_different_sections_apart():
    # GAINTABLE is used by several sections; the section key separates them
    labels = [
        QASection.FLAGGING.GAINTABLE,
        QASection.BANDPASS.GAINTABLE,
        QASection.DELAY.GAINTABLE,
    ]

    assert {label.section for label in labels} == {
        "flagging",
        "bandpass",
        "delay",
    }


def test_should_attach_spec_to_label():
    assert QASection.METADATA_SUMMARY.CHANNELS.spec == LabelSpec(
        Level.FIELD_SINGLE, int
    )
    assert QASection.METADATA_SUMMARY.TIMESTAMPS.spec == LabelSpec(
        Level.FIELD_MULTIPLE, list[float]
    )
    assert QASection.FLAGGING.PRE_FLAGGING.spec == LabelSpec(
        Level.PREFIX_SINGLE, str
    )
    assert QASection.BANDPASS.GAINTABLE.spec == LabelSpec(
        Level.PREFIX_SINGLE, str
    )


def test_should_look_up_label_by_its_string_value():
    # Enum lookup by value does not call __new__, so spec is not needed;
    # pylint cannot tell and reads it as a constructor call
    # pylint: disable-next=no-value-for-parameter
    label = QASection.METADATA_SUMMARY("channels")

    assert label is QASection.METADATA_SUMMARY.CHANNELS


def test_should_overwrite_field_single_label():
    section = {}
    label = QASection.METADATA_SUMMARY.CHANNELS

    label.store(section, 10)
    label.store(section, 20)

    assert label.load(section) == 20


def test_should_append_field_multiple_label_in_order():
    section = {}
    label = QASection.METADATA_SUMMARY.MS

    label.store(section, "a.ms")
    label.store(section, "b.ms")

    assert label.load(section) == ["a.ms", "b.ms"]


def test_should_overwrite_prefix_single_label_per_prefix():
    section = {}
    label = QASection.FLAGGING.PRE_FLAGGING

    label.store(section, "a_old.json", prefix="a")
    label.store(section, "a.json", prefix="a")
    label.store(section, "b.json", prefix="b")

    assert label.load(section) == {"a": "a.json", "b": "b.json"}


def test_should_load_none_for_unset_label():
    assert QASection.METADATA_SUMMARY.CHANNELS.load({}) is None
    assert QASection.BANDPASS.GAINTABLE.load({}) is None


def test_should_keep_labels_of_same_name_apart_within_a_section():
    # Sections are stored separately, so labels sharing a value do not
    # collide as long as each section has its own dict
    flagging, bandpass = {}, {}

    QASection.FLAGGING.GAINTABLE.store(flagging, "flag.h5", prefix="a")
    QASection.BANDPASS.GAINTABLE.store(bandpass, "bp.h5", prefix="a")

    assert flagging == {"gaintable": {"a": "flag.h5"}}
    assert bandpass == {"gaintable": {"a": "bp.h5"}}


@pytest.mark.parametrize(
    "label",
    [QASection.FLAGGING.PRE_FLAGGING, QASection.DELAY.DELAYTABLE],
)
def test_should_require_prefix_for_prefix_level_label(label):
    with pytest.raises(RunMetadataError, match="needs a prefix"):
        label.store({}, "data.h5")


@pytest.mark.parametrize(
    "label",
    [QASection.METADATA_SUMMARY.CHANNELS, QASection.METADATA_SUMMARY.MS],
)
def test_should_reject_prefix_for_field_level_label(label):
    with pytest.raises(RunMetadataError, match="takes no prefix"):
        label.store({}, "a.ms", prefix="a")


@pytest.mark.parametrize(
    "label, data",
    [
        (QASection.METADATA_SUMMARY.CHANNELS, "4096"),
        (QASection.METADATA_SUMMARY.TIMESTAMPS, 1.0),
        (QASection.METADATA_SUMMARY.MS, 1),
    ],
)
def test_should_reject_entry_not_matching_schema(label, data):
    with pytest.raises(RunMetadataError, match="expects"):
        label.store({}, data)


def test_should_check_typeddict_schema_as_dict():
    class Entry(TypedDict):
        path: str

    # Label defined for this test only, with a TypedDict schema
    class EXAMPLE(QASectionLabel):
        ENTRY = "entry", LabelSpec(Level.FIELD_MULTIPLE, Entry)

    section = {}
    EXAMPLE.ENTRY.store(section, {"path": "a.ms"})

    assert EXAMPLE.ENTRY.load(section) == [{"path": "a.ms"}]
    with pytest.raises(RunMetadataError, match="expects dict"):
        EXAMPLE.ENTRY.store(section, "a.ms")
