import pytest
from mock import patch

from ska_sdp_instrumental_calibration.qa_metrics.report import (
    REPORT_SECTIONS,
    generate_qa_metric,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.qa_section import (
    QASection,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.run_metadata import (  # noqa: E501
    RunMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.view.section_builder import (
    section_builder,
)
from ska_sdp_instrumental_calibration.qa_metrics.view.sections import (
    BandpassFitSection,
    CableDelaysSection,
    FlaggingStatisticsSection,
    MetadataSection,
)

# Report of the whole run, written when no field report can be built: it
# has no field name, so it cannot overwrite the report of a field
ERROR_REPORT = "inst-qa-metric.html"
BUILD = (
    "ska_sdp_instrumental_calibration.qa_metrics.report"
    ".section_builder.build"
)
THRESHOLDS = {"phase_fit_threshold": 0.75, "gain_fit_threshold": 0.75}
LOGGER = "ska_sdp_instrumental_calibration.qa_metrics.report.logger"


def _write_run_metadata(qa_dir, field_names):
    """
    Run metadata of fields with only their measurement set recorded: their
    products are missing, so each section of their report is an error
    """
    run_metadata = RunMetadata(str(qa_dir))
    for field_name in field_names:
        run_metadata.add_for_field(
            field_name, QASection.METADATA_SUMMARY.MS, f"/data/{field_name}.ms"
        )
    run_metadata.to_json()


def test_should_show_sections_in_report_order():
    assert REPORT_SECTIONS == (
        MetadataSection,
        FlaggingStatisticsSection,
        BandpassFitSection,
        CableDelaysSection,
    )


def test_should_have_a_builder_for_every_report_section():
    # A section added to the report without a builder would fail every run
    # pylint: disable-next=protected-access
    assert set(REPORT_SECTIONS) <= set(section_builder._builders)


def test_should_write_one_report_per_field(tmp_path):
    _write_run_metadata(tmp_path, ["field_a", "field_b"])

    generate_qa_metric(str(tmp_path), **THRESHOLDS)

    assert (tmp_path / "inst-field_a-qa-metric.html").exists()
    assert (tmp_path / "inst-field_b-qa-metric.html").exists()
    assert not (tmp_path / ERROR_REPORT).exists()


def test_should_show_missing_products_in_their_sections(tmp_path):
    # The field has no gaintable, pre-flagging data or delay table: the
    # report is still written, with an error in each section
    _write_run_metadata(tmp_path, ["field_a"])

    generate_qa_metric(str(tmp_path), title="Test QA Metrics", **THRESHOLDS)

    page = (tmp_path / "inst-field_a-qa-metric.html").read_text()
    assert "Test QA Metrics" in page
    assert "Source: <strong>field_a</strong>" in page
    for section_type in REPORT_SECTIONS:
        assert f"{section_type.title} of field field_a cannot be built" in page


@patch(BUILD, wraps=section_builder.build)
def test_should_build_report_sections_in_order_for_each_field(
    build_mock, tmp_path
):
    _write_run_metadata(tmp_path, ["field_a"])

    generate_qa_metric(str(tmp_path), **THRESHOLDS)

    built = [args[0] for args, _ in build_mock.call_args_list]
    assert built == list(REPORT_SECTIONS)


def test_should_write_error_report_if_no_run_metadata(tmp_path):
    with patch(LOGGER) as logger_mock:
        generate_qa_metric(str(tmp_path), **THRESHOLDS)

    page = (tmp_path / ERROR_REPORT).read_text()
    assert f"QA metric report of the INST run in {tmp_path}" in page
    assert f"No run metadata found in {tmp_path}" in page
    # The report of the run has no source
    assert "Source:" not in page
    # A run metadata error is expected: logged without its traceback
    logger_mock.warning.assert_called_once()
    assert logger_mock.warning.call_args.kwargs == {"exc_info": False}


def test_should_not_overwrite_report_of_a_field_named_null(tmp_path):
    # A field can be named NULL by upstream errors: its report is its own
    _write_run_metadata(tmp_path, ["NULL"])

    generate_qa_metric(str(tmp_path), **THRESHOLDS)

    page = (tmp_path / "inst-NULL-qa-metric.html").read_text()
    assert "Source: <strong>NULL</strong>" in page
    assert not (tmp_path / ERROR_REPORT).exists()


@pytest.mark.parametrize(
    "run_metadata, cause",
    [
        # Truncated, e.g. by a failed write
        ('{"field_a": {', "JSONDecodeError"),
        # A field of an unexpected shape
        ('{"field_a": {"unexpected": 1}}', "TypeError"),
    ],
)
def test_should_write_error_report_if_run_metadata_is_corrupt(
    tmp_path, run_metadata, cause
):
    (tmp_path / "run_metadata.json").write_text(run_metadata)

    with patch(LOGGER) as logger_mock:
        generate_qa_metric(str(tmp_path), **THRESHOLDS)

    page = (tmp_path / ERROR_REPORT).read_text()
    assert f"QA metric report of the INST run in {tmp_path}" in page
    assert f"Caused by {cause}" in page
    # An unexpected error is logged with its traceback, to trace it
    assert logger_mock.warning.call_args.kwargs == {"exc_info": True}


def test_should_report_other_fields_if_one_fails(tmp_path):
    # A bug in the report of one field is logged, and the other fields are
    # still reported
    _write_run_metadata(tmp_path, ["field_a", "field_b"])
    # pylint: disable-next=protected-access
    metadata_builder = section_builder._builders[MetadataSection]

    def _buggy_metadata_section(field_metadata, **kwargs):
        if field_metadata.field_name == "field_a":
            raise ZeroDivisionError("bug in the computation")
        return metadata_builder(field_metadata, **kwargs)

    with patch.dict(
        # pylint: disable-next=protected-access
        section_builder._builders,
        {MetadataSection: _buggy_metadata_section},
    ), patch(LOGGER) as logger_mock:
        generate_qa_metric(str(tmp_path), **THRESHOLDS)

    assert not (tmp_path / "inst-field_a-qa-metric.html").exists()
    assert (tmp_path / "inst-field_b-qa-metric.html").exists()
    logger_mock.exception.assert_called_once_with(
        "QA metric report of field %s cannot be written", "field_a"
    )


def test_should_write_error_report_if_no_field_is_recorded(tmp_path):
    # e.g. the run failed before loading data
    RunMetadata(str(tmp_path)).to_json()

    with patch(LOGGER) as logger_mock:
        generate_qa_metric(str(tmp_path), **THRESHOLDS)

    page = (tmp_path / ERROR_REPORT).read_text()
    assert f"QA metric report of the INST run in {tmp_path}" in page
    assert f"No field recorded in the run metadata of {tmp_path}" in page
    # The reason is also logged
    (_, logged_error), _ = logger_mock.warning.call_args
    assert str(logged_error) == (
        f"No field recorded in the run metadata of {tmp_path}"
    )
