import pytest

from ska_sdp_instrumental_calibration.data_managers.data_export.qa_metadata import (  # noqa: E501
    export_pre_flagging,
)


def test_should_export_pre_flags():
    with pytest.raises(NotImplementedError):
        export_pre_flagging("flags", "path/to/output")
