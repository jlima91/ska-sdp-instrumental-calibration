from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.service.cable_delays_metric import (  # noqa: E501
    get_delaytable_paths,
)


def test_should_get_delaytable_of_each_ms():
    field_metadata = FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "delay": {
                "delaytable": {
                    "ms_a": "/qa/dir/ms_a-delay.h5parm",
                    "ms_b": "/qa/dir/ms_b-delay.h5parm",
                }
            }
        },
    )

    assert get_delaytable_paths(field_metadata) == [
        "/qa/dir/ms_a-delay.h5parm",
        "/qa/dir/ms_b-delay.h5parm",
    ]


def test_should_get_no_delaytable_if_none_is_recorded():
    # Not an error here: the reader raises for an empty list
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    assert get_delaytable_paths(field_metadata) == []
