import json

import numpy as np
import pytest
from mock import Mock, patch

from ska_sdp_instrumental_calibration.qa_metrics import field_sections
from ska_sdp_instrumental_calibration.qa_metrics.domain.qa_metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.products import read_gaintable
from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.sections import (
    ErrorSection,
    FlaggingStatisticsSection,
    MetadataSection,
)
from ska_sdp_instrumental_calibration.qa_metrics.tables import (
    FLAGGING_STATISTICS_COLUMNS,
)

READ_H5PARM_GAINS = (
    "ska_sdp_instrumental_calibration.qa_metrics.products.gaintable"
    ".read_h5parm_gains"
)
GAINTABLE = "/qa/dir/field_a.h5parm"


@pytest.fixture(autouse=True)
def clear_gaintable_cache():
    """
    Tests mock read_h5parm_gains differently for the same gaintable path,
    so a gaintable read cached by one test must not leak into the next
    """
    read_gaintable.cache_clear()
    yield
    read_gaintable.cache_clear()


def _field_metadata(metadata_summary):
    return FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={"metadata_summary": metadata_summary},
    )


def _gains_with_axes(frequency, antenna):
    soltab = Mock(name="soltab", frequency=frequency, antenna=antenna)
    return "gain", "weight", "solved", soltab


@patch(READ_H5PARM_GAINS)
def test_should_get_metadata_section_of_a_field(read_h5parm_gains_mock):
    read_h5parm_gains_mock.return_value = _gains_with_axes(
        np.linspace(1e8, 2e8, 64), ["station_a", "station_b"]
    )
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0, 30.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    assert field_sections.build(
        MetadataSection, field_metadata
    ) == MetadataSection(
        file_information=FileInformation(
            input_ms=["/data/a.ms"], output_path="/qa/dir"
        ),
        observation_parameters=ObservationParameters(
            start_time=5.0,
            observation_length=30.0,
            integration_time=10.0,
            channels=64,
        ),
        station_configuration=StationConfiguration(
            {"0": "station_a", "1": "station_b"}
        ),
    )
    read_h5parm_gains_mock.assert_called_once_with(GAINTABLE)


@patch(READ_H5PARM_GAINS)
def test_should_join_observation_parameters_of_all_ms_along_time(
    read_h5parm_gains_mock,
):
    read_h5parm_gains_mock.return_value = _gains_with_axes(
        np.linspace(1e8, 2e8, 64), ["station_a"]
    )
    # MSs are not recorded in time order
    field_metadata = _field_metadata(
        {
            "MS": ["/data/b.ms", "/data/a.ms"],
            "timestamps": [[40.0, 50.0], [10.0, 20.0, 30.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    qa_metadata = field_sections.build(MetadataSection, field_metadata)

    assert qa_metadata.observation_parameters == ObservationParameters(
        start_time=5.0,
        observation_length=50.0,
        integration_time=10.0,
        channels=64,
    )
    assert qa_metadata.file_information.input_ms == [
        "/data/b.ms",
        "/data/a.ms",
    ]


@patch(READ_H5PARM_GAINS)
def test_should_leave_channels_unset_if_not_recorded(read_h5parm_gains_mock):
    # Missing channels must not stop the report; it shows them as unknown
    read_h5parm_gains_mock.return_value = _gains_with_axes(
        np.linspace(1e8, 2e8, 64), ["station_a"]
    )
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0, 30.0]],
            "final_gaintable": GAINTABLE,
        }
    )

    qa_metadata = field_sections.build(MetadataSection, field_metadata)

    assert qa_metadata.observation_parameters.channels is None


def _assert_metadata_error_section(section, cause_match):
    """
    The section is the error section of the metadata summary of field_a,
    caused by a RunMetadataError matching cause_match
    """
    assert isinstance(section, ErrorSection)
    assert section.title == MetadataSection.title
    assert "field_a" in str(section.error)
    assert isinstance(section.error.__cause__, RunMetadataError)
    assert cause_match in str(section.error.__cause__)


@pytest.mark.parametrize(
    "metadata_summary",
    [
        # Each case misses exactly one of the required labels; channels
        # is optional, so it is recorded in every case
        {
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        },
        {"MS": ["/data/a.ms"], "final_gaintable": GAINTABLE, "channels": 64},
        {"MS": ["/data/a.ms"], "timestamps": [[10.0, 20.0]], "channels": 64},
        {
            "MS": [],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        },
        {},
    ],
)
@patch(READ_H5PARM_GAINS)
def test_should_get_error_section_if_qa_metadata_is_missing(
    read_h5parm_gains_mock, metadata_summary
):
    field_metadata = _field_metadata(metadata_summary)

    section = field_sections.build(MetadataSection, field_metadata)

    _assert_metadata_error_section(section, cause_match="field_a")
    read_h5parm_gains_mock.assert_not_called()


@patch(READ_H5PARM_GAINS)
def test_should_get_error_section_if_gaintable_has_no_antenna_axis(
    read_h5parm_gains_mock,
):
    # Channels come from the run metadata, so only the antenna axis is
    # needed off the gaintable
    read_h5parm_gains_mock.return_value = _gains_with_axes(
        np.linspace(1e8, 2e8, 64), None
    )
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    section = field_sections.build(MetadataSection, field_metadata)

    _assert_metadata_error_section(section, cause_match="no antenna axis")


@patch(READ_H5PARM_GAINS)
def test_should_get_error_section_if_final_gaintable_was_not_produced(
    read_h5parm_gains_mock,
):
    read_h5parm_gains_mock.side_effect = FileNotFoundError(GAINTABLE)
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    section = field_sections.build(MetadataSection, field_metadata)

    _assert_metadata_error_section(section, cause_match="was not produced")


def _flagging_field_metadata(pre_flagging):
    return FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "metadata_summary": {"final_gaintable": GAINTABLE},
            "flagging": {"pre_flagging": pre_flagging},
        },
    )


def _write_pre_flagging(path, unsolvable, nchannels):
    nstations = len(unsolvable)
    path.write_text(
        json.dumps(
            {
                "flagged": [0] * nstations,
                "total": [1] * nstations,
                "unsolvable": unsolvable,
                "nchannels": nchannels,
            }
        )
    )
    return str(path)


@patch(READ_H5PARM_GAINS)
def test_should_get_flagging_statistics_section_of_a_field(
    read_h5parm_gains_mock, tmp_path
):
    # (time, antenna, frequency, 2, 2): 1 time, 2 stations, 2 channels
    weight = np.ones((1, 2, 2, 2, 2))
    # station_a: 1 of its 4 parallel hand gains flagged after calibration
    weight[0, 0, 0, 0, 0] = 0
    read_h5parm_gains_mock.return_value = (
        "gain",
        weight,
        "solved",
        Mock(name="soltab", antenna=["station_a", "station_b"]),
    )
    # station_b: 1 of its 2 channels cannot be solved before calibration
    field_metadata = _flagging_field_metadata(
        {
            "ms_a": _write_pre_flagging(
                tmp_path / "a.json", unsolvable=[0, 1], nchannels=2
            )
        }
    )

    section = field_sections.build(FlaggingStatisticsSection, field_metadata)

    assert section == FlaggingStatisticsSection(
        columns=FLAGGING_STATISTICS_COLUMNS,
        rows=[
            {
                "station_id": "0",
                "station_name": "station_a",
                "flagged_before": 0.0,
                "flagged_after": 25.0,
                "flagged_additional": 25.0,
            },
            {
                "station_id": "1",
                "station_name": "station_b",
                "flagged_before": 50.0,
                "flagged_after": 0.0,
                "flagged_additional": -50.0,
            },
        ],
        totals={"flagged_before": 25.0, "flagged_after": 12.5},
    )
    # Stations and weights come from a single read of the gaintable
    read_h5parm_gains_mock.assert_called_once_with(GAINTABLE)


@patch(READ_H5PARM_GAINS)
def test_should_get_error_section_if_pre_flagging_is_missing(
    read_h5parm_gains_mock, tmp_path
):
    read_h5parm_gains_mock.return_value = _gains_with_axes(
        np.linspace(1e8, 2e8, 2), ["station_a"]
    )
    field_metadata = _flagging_field_metadata(
        {"ms_a": str(tmp_path / "missing.json")}
    )

    section = field_sections.build(FlaggingStatisticsSection, field_metadata)

    assert isinstance(section, ErrorSection)
    assert section.title == FlaggingStatisticsSection.title
    assert "field_a" in str(section.error)
    assert "was not produced" in str(section.error.__cause__)


@patch(READ_H5PARM_GAINS)
def test_should_get_error_section_if_final_gaintable_is_not_recorded(
    read_h5parm_gains_mock,
):
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    section = field_sections.build(FlaggingStatisticsSection, field_metadata)

    assert isinstance(section, ErrorSection)
    assert "No final gaintable" in str(section.error.__cause__)
    read_h5parm_gains_mock.assert_not_called()


SECTION_BUILDERS = (
    "ska_sdp_instrumental_calibration.qa_metrics.field_sections"
    "._SECTION_BUILDERS"
)


def test_should_raise_errors_other_than_run_metadata_errors():
    # Any other error is a bug: it must fail loudly, not hide in the report
    builder = Mock(side_effect=ValueError("bug in the computation"))
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    with patch.dict(SECTION_BUILDERS, {MetadataSection: builder}):
        with pytest.raises(ValueError, match="bug in the computation"):
            field_sections.build(MetadataSection, field_metadata)


def test_should_raise_if_no_builder_for_section_type():
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    with pytest.raises(KeyError):
        field_sections.build(ErrorSection, field_metadata)


@pytest.mark.parametrize(
    "section_type", [MetadataSection, FlaggingStatisticsSection]
)
@patch(READ_H5PARM_GAINS)
def test_should_get_error_section_if_final_gaintable_cannot_be_read(
    read_h5parm_gains_mock, section_type, tmp_path
):
    # A corrupt gaintable is a bad product: it is shown in the section
    # rather than failing the report
    read_h5parm_gains_mock.side_effect = OSError("file signature not found")
    field_metadata = FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "metadata_summary": {
                "MS": ["/data/a.ms"],
                "timestamps": [[10.0, 20.0]],
                "final_gaintable": GAINTABLE,
            },
            "flagging": {
                "pre_flagging": {
                    "ms_a": _write_pre_flagging(
                        tmp_path / "a.json", unsolvable=[0], nchannels=1
                    )
                }
            },
        },
    )

    section = field_sections.build(section_type, field_metadata)

    assert isinstance(section, ErrorSection)
    assert section.title == section_type.title
    assert "cannot be read" in str(section.error.__cause__)
