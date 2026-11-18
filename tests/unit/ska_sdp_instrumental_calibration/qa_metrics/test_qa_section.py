from ska_sdp_instrumental_calibration.qa_metrics.qa_section import QASection


def test_should_derive_section_key_from_class_name():
    assert QASection.METADATA_SUMMARY.MS.section == "metadata_summary"
    assert (
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA.section
        == "metadata_summary"
    )
    assert QASection.FLAGGING.GAIN_FITS.section == "flagging"
    assert QASection.FLAGGING.GAINTABLE.section == "flagging"
    assert QASection.BANDPASS.GAINTABLE.section == "bandpass"
    assert QASection.DELAY.GAINTABLE.section == "delay"
    assert QASection.DELAY.DELAYTABLE.section == "delay"


def test_should_use_label_value_as_plain_string():
    assert QASection.METADATA_SUMMARY.MS == "MS"
    assert (
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA == "visibility_metadata"
    )
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
