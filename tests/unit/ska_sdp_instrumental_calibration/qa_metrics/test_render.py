from ska_sdp_instrumental_calibration.qa_metrics.render import render_page


def test_should_render_page_with_title_and_section():
    page = render_page("Title", ["Section"])

    assert "<title>Title</title>" in page
    assert "<summary>Section</summary>" in page
