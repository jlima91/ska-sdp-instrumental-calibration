import pytest

from ska_sdp_instrumental_calibration.data_managers.processing_set import (
    field_name_without_xradio_suffix,
)


@pytest.mark.parametrize(
    "field_name, expected",
    [
        ("CAL_FIELD_0", "CAL_FIELD"),
        ("3C286_12", "3C286"),
        # Only the last suffix is the field id added by xradio
        ("FIELD_2_3", "FIELD_2"),
        ("J1939-6342_1", "J1939-6342"),
        # Unnamed MSv2 field
        ("_0", "UNKNOWN_FIELD"),
        ("NO_SUFFIX", "NO_SUFFIX"),
    ],
)
def test_should_remove_xradio_field_id_suffix(field_name, expected):
    assert field_name_without_xradio_suffix(field_name) == expected
