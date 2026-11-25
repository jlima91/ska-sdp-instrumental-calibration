import pytest

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.run_metadata import (  # noqa: E501
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.service.flagging_statistics_metric import (  # noqa: E501
    get_final_gaintable,
    get_pre_flagging_paths,
)


def _field_metadata(sections):
    return FieldMetadata("field_a", "/qa/dir", sections=sections)


def test_should_get_final_gaintable_of_a_field():
    field_metadata = _field_metadata(
        {"metadata_summary": {"final_gaintable": "/qa/dir/field_a.h5parm"}}
    )

    assert get_final_gaintable(field_metadata) == "/qa/dir/field_a.h5parm"


@pytest.mark.parametrize(
    "sections",
    [
        {},
        {"metadata_summary": {}},
        {"metadata_summary": {"final_gaintable": ""}},
    ],
)
def test_should_raise_if_no_final_gaintable_is_recorded(sections):
    with pytest.raises(
        RunMetadataError, match="No final gaintable recorded for field field_a"
    ):
        get_final_gaintable(_field_metadata(sections))


def test_should_get_pre_flagging_data_of_each_ms():
    field_metadata = _field_metadata(
        {
            "flagging": {
                "pre_flagging": {
                    "ms_a": "/qa/dir/derived/ms_a/pre-flagging-data.json",
                    "ms_b": "/qa/dir/derived/ms_b/pre-flagging-data.json",
                }
            }
        }
    )

    assert get_pre_flagging_paths(field_metadata) == [
        "/qa/dir/derived/ms_a/pre-flagging-data.json",
        "/qa/dir/derived/ms_b/pre-flagging-data.json",
    ]


def test_should_get_no_pre_flagging_data_if_none_is_recorded():
    # Not an error here: the reader raises for an empty list
    assert get_pre_flagging_paths(_field_metadata({})) == []
