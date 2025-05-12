from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput


def test_should_return_manage_stage_count():
    upstream_output = UpstreamOutput()

    assert upstream_output.get_call_count("stage1") == 0
    upstream_output.increment_call_count("stage1")
    assert upstream_output.get_call_count("stage1") == 1
    upstream_output.increment_call_count("stage1")
    assert upstream_output.get_call_count("stage1") == 2

    assert upstream_output.get_call_count("stage2") == 0
    upstream_output.increment_call_count("stage2")
    assert upstream_output.get_call_count("stage2") == 1
    upstream_output.increment_call_count("stage2")
    assert upstream_output.get_call_count("stage2") == 2
