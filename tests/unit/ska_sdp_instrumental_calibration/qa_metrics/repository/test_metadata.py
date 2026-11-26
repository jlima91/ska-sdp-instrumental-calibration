import pytest

from ska_sdp_instrumental_calibration.qa_metrics.domain.metadata import (
    StationConfiguration,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.field_metadata import (  # noqa: E501
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.metadata import (
    get_channels,
    get_station_configuration,
    get_timestamps,
)


def test_should_get_station_configuration_of_a_field():
    # As load_data records them: MS station ID keyed by MS station name
    stations = {"station_a": "0", "station_b": "1"}
    field_metadata = FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={"metadata_summary": {"stations": stations}},
    )

    assert get_station_configuration(field_metadata) == StationConfiguration(
        stations
    )


@pytest.mark.parametrize(
    "sections",
    [{}, {"metadata_summary": {}}, {"metadata_summary": {"stations": {}}}],
)
def test_should_get_no_stations_if_not_recorded(sections):
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections=sections)

    assert get_station_configuration(field_metadata) is None


def _metadata_summary(metadata_summary):
    return FieldMetadata(
        "field_a", "/qa/dir", sections={"metadata_summary": metadata_summary}
    )


def test_should_get_timestamps_of_each_ms():
    timestamps = [[10.0, 20.0], [30.0, 40.0]]

    assert (
        get_timestamps(_metadata_summary({"timestamps": timestamps}))
        == timestamps
    )


@pytest.mark.parametrize("metadata_summary", [{}, {"timestamps": []}])
def test_should_get_no_timestamps_if_not_recorded(metadata_summary):
    assert get_timestamps(_metadata_summary(metadata_summary)) is None


def test_should_get_channels():
    assert get_channels(_metadata_summary({"channels": 64})) == 64


def test_should_get_no_channels_if_not_recorded():
    # Not an error: the report shows the channels as unknown
    assert get_channels(_metadata_summary({})) is None
