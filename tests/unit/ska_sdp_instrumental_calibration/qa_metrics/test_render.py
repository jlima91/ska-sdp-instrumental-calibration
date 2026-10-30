from ska_sdp_instrumental_calibration.qa_metrics.render import (
    render_page,
    render_section,
)


def test_should_render_section_with_title():
    section = render_section("Metadata")

    assert "<summary>Metadata</summary>" in section


def test_should_render_page_with_title_and_sections():
    section = render_section("Metadata")

    page = render_page("Bandpass QA Metrics", [section])

    assert "Bandpass QA Metrics" in page
    assert section in page
