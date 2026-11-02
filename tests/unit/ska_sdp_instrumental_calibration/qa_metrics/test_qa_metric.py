from ska_sdp_instrumental_calibration.qa_metrics.qa_metric import (
    PAGE_TITLE,
    SECTION_TITLES,
    generate_qa_metric,
    write_qa_metric,
)


def test_should_write_qa_metric_page_named_after_source_name(tmp_path):
    write_qa_metric("<html></html>", str(tmp_path), "source_name")

    assert (
        tmp_path / "inst-source_name-qa-metric.html"
    ).read_text() == "<html></html>"


def test_should_generate_qa_metric_in_output_path_without_sdm(tmp_path):
    (tmp_path / "source_name_gaintable.h5parm").touch()

    generate_qa_metric(str(tmp_path))

    page = (tmp_path / "inst-source_name-qa-metric.html").read_text()
    assert PAGE_TITLE in page
    assert all(title in page for title in SECTION_TITLES)
