from ska_sdp_instrumental_calibration.tagger import Tagger


def test_should_tag_function():
    _test_tag = Tagger("TEST")

    @_test_tag
    def func():
        pass

    assert func in _test_tag
