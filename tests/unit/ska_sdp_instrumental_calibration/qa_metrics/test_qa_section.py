from ska_sdp_instrumental_calibration.qa_metrics.qa_section import QASection


def test_should_derive_section_key_from_class_name():
    assert QASection.METADATA_SUMMARY.MS.section == "metadata_summary"
    assert (
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA.section
        == "metadata_summary"
    )
    assert QASection.FLAGGING.GAIN_FITS.section == "flagging"


def test_should_use_label_value_as_plain_string():
    assert QASection.METADATA_SUMMARY.MS == "MS"
    assert (
        QASection.METADATA_SUMMARY.VISIBILITY_METADATA == "visibility_metadata"
    )
    assert QASection.FLAGGING.GAIN_FITS == "gain_fits"
