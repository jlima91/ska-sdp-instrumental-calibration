import numpy as np
import pytest
from mock import Mock, patch

from ska_sdp_instrumental_calibration.exceptions import QAMetadataError
from ska_sdp_instrumental_calibration.qa_metrics.domain import QAMetadata
from ska_sdp_instrumental_calibration.qa_metrics.domain.qa_metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from ska_sdp_instrumental_calibration.qa_metrics.field_queries import (
    get_qa_metadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
)

READ_H5PARM_GAINS = (
    "ska_sdp_instrumental_calibration.qa_metrics.field_queries"
    ".read_h5parm_gains"
)
GAINTABLE = "/qa/dir/field_a.h5parm"


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
def test_should_get_qa_metadata_of_a_field(read_h5parm_gains_mock):
    read_h5parm_gains_mock.return_value = _gains_with_axes(
        np.linspace(1e8, 2e8, 64), ["station_a", "station_b"]
    )
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0, 30.0]],
            "final_gaintable": [GAINTABLE],
        }
    )

    assert get_qa_metadata(field_metadata) == QAMetadata(
        source_name="field_a",
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
            "final_gaintable": [GAINTABLE],
        }
    )

    qa_metadata = get_qa_metadata(field_metadata)

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


@pytest.mark.parametrize(
    "metadata_summary",
    [
        {"timestamps": [[10.0, 20.0]], "final_gaintable": [GAINTABLE]},
        {"MS": ["/data/a.ms"], "final_gaintable": [GAINTABLE]},
        {"MS": ["/data/a.ms"], "timestamps": [[10.0, 20.0]]},
        {
            "MS": [],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": [GAINTABLE],
        },
        {},
    ],
)
@patch(READ_H5PARM_GAINS)
def test_should_raise_if_qa_metadata_is_missing(
    read_h5parm_gains_mock, metadata_summary
):
    field_metadata = _field_metadata(metadata_summary)

    with pytest.raises(QAMetadataError, match="field_a"):
        get_qa_metadata(field_metadata)

    read_h5parm_gains_mock.assert_not_called()


@pytest.mark.parametrize(
    "frequency, antenna",
    [(None, ["station_a"]), (np.linspace(1e8, 2e8, 64), None)],
)
@patch(READ_H5PARM_GAINS)
def test_should_raise_if_gaintable_has_no_frequency_or_antenna_axis(
    read_h5parm_gains_mock, frequency, antenna
):
    read_h5parm_gains_mock.return_value = _gains_with_axes(frequency, antenna)
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": [GAINTABLE],
        }
    )

    with pytest.raises(QAMetadataError, match="no frequency or antenna"):
        get_qa_metadata(field_metadata)


@patch(READ_H5PARM_GAINS)
def test_should_raise_if_final_gaintable_was_not_produced(
    read_h5parm_gains_mock,
):
    read_h5parm_gains_mock.side_effect = FileNotFoundError(GAINTABLE)
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": [GAINTABLE],
        }
    )

    with pytest.raises(QAMetadataError, match="was not produced"):
        get_qa_metadata(field_metadata)
