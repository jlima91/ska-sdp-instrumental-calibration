from typing import TypedDict

import pytest

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.manifest_schema import (  # noqa: E501
    LabelSpec,
    Level,
    ManifestSchema,
    QASectionLabel,
)


def test_should_derive_section_key_from_class_name():
    assert ManifestSchema.METADATA_SUMMARY.MS.section == "metadata_summary"
    assert (
        ManifestSchema.METADATA_SUMMARY.TIMESTAMPS.section
        == "metadata_summary"
    )
    assert (
        ManifestSchema.METADATA_SUMMARY.FINAL_GAINTABLE.section
        == "metadata_summary"
    )
    assert ManifestSchema.FLAGGING.GAIN_FITS.section == "flagging"
    assert ManifestSchema.FLAGGING.GAINTABLE.section == "flagging"
    assert ManifestSchema.BANDPASS.GAINTABLE.section == "bandpass"
    assert ManifestSchema.DELAY.GAINTABLE.section == "delay"
    assert ManifestSchema.DELAY.DELAYTABLE.section == "delay"


def test_should_use_label_value_as_plain_string():
    assert ManifestSchema.METADATA_SUMMARY.MS == "MS"
    assert ManifestSchema.METADATA_SUMMARY.TIMESTAMPS == "timestamps"
    assert ManifestSchema.METADATA_SUMMARY.FINAL_GAINTABLE == "final_gaintable"
    assert ManifestSchema.FLAGGING.GAIN_FITS == "gain_fits"
    assert ManifestSchema.FLAGGING.GAINTABLE == "gaintable"
    assert ManifestSchema.BANDPASS.GAINTABLE == "gaintable"
    assert ManifestSchema.DELAY.GAINTABLE == "gaintable"
    assert ManifestSchema.DELAY.DELAYTABLE == "delaytable"


def test_should_keep_same_label_of_different_sections_apart():
    # GAINTABLE is used by several sections; the section key separates them
    labels = [
        ManifestSchema.FLAGGING.GAINTABLE,
        ManifestSchema.BANDPASS.GAINTABLE,
        ManifestSchema.DELAY.GAINTABLE,
    ]

    assert {label.section for label in labels} == {
        "flagging",
        "bandpass",
        "delay",
    }


def test_should_attach_spec_to_label():
    assert ManifestSchema.METADATA_SUMMARY.CHANNELS.spec == LabelSpec(
        Level.FIELD_SINGLE, int
    )
    assert ManifestSchema.METADATA_SUMMARY.TIMESTAMPS.spec == LabelSpec(
        Level.FIELD_MULTIPLE, list[float]
    )
    assert ManifestSchema.FLAGGING.PRE_FLAGGING.spec == LabelSpec(
        Level.PREFIX_SINGLE, str
    )
    assert ManifestSchema.BANDPASS.GAINTABLE.spec == LabelSpec(
        Level.PREFIX_SINGLE, str
    )


def test_should_look_up_label_by_its_string_value():
    # Enum lookup by value does not call __new__, so spec is not needed;
    # pylint cannot tell and reads it as a constructor call
    # pylint: disable-next=no-value-for-parameter
    label = ManifestSchema.METADATA_SUMMARY("channels")

    assert label is ManifestSchema.METADATA_SUMMARY.CHANNELS


def test_should_overwrite_field_single_label():
    section = {}
    label = ManifestSchema.METADATA_SUMMARY.CHANNELS

    label.store(section, 10)
    label.store(section, 20)

    assert label.load(section) == 20


def test_should_append_field_multiple_label_in_order():
    section = {}
    label = ManifestSchema.METADATA_SUMMARY.MS

    label.store(section, "a.ms")
    label.store(section, "b.ms")

    assert label.load(section) == ["a.ms", "b.ms"]


def test_should_overwrite_prefix_single_label_per_prefix():
    section = {}
    label = ManifestSchema.FLAGGING.PRE_FLAGGING

    label.store(section, "a_old.json", prefix="a")
    label.store(section, "a.json", prefix="a")
    label.store(section, "b.json", prefix="b")

    assert label.load(section) == {"a": "a.json", "b": "b.json"}


def test_should_load_none_for_unset_label():
    assert ManifestSchema.METADATA_SUMMARY.CHANNELS.load({}) is None
    assert ManifestSchema.BANDPASS.GAINTABLE.load({}) is None


def test_should_keep_labels_of_same_name_apart_within_a_section():
    # Sections are stored separately, so labels sharing a value do not
    # collide as long as each section has its own dict
    flagging, bandpass = {}, {}

    ManifestSchema.FLAGGING.GAINTABLE.store(flagging, "flag.h5", prefix="a")
    ManifestSchema.BANDPASS.GAINTABLE.store(bandpass, "bp.h5", prefix="a")

    assert flagging == {"gaintable": {"a": "flag.h5"}}
    assert bandpass == {"gaintable": {"a": "bp.h5"}}


@pytest.mark.parametrize(
    "label",
    [ManifestSchema.FLAGGING.PRE_FLAGGING, ManifestSchema.DELAY.DELAYTABLE],
)
def test_should_require_prefix_for_prefix_level_label(label):
    with pytest.raises(RunMetadataError, match="needs a prefix"):
        label.store({}, "data.h5")


@pytest.mark.parametrize(
    "label",
    [
        ManifestSchema.METADATA_SUMMARY.CHANNELS,
        ManifestSchema.METADATA_SUMMARY.MS,
    ],
)
def test_should_reject_prefix_for_field_level_label(label):
    with pytest.raises(RunMetadataError, match="takes no prefix"):
        label.store({}, "a.ms", prefix="a")


@pytest.mark.parametrize(
    "label, data",
    [
        (ManifestSchema.METADATA_SUMMARY.CHANNELS, "4096"),
        (ManifestSchema.METADATA_SUMMARY.TIMESTAMPS, 1.0),
        (ManifestSchema.METADATA_SUMMARY.MS, 1),
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
