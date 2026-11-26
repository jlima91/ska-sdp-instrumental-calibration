import numpy as np
import pytest

from ska_sdp_instrumental_calibration.qa_metrics.domain.metadata import (
    ObservationParameters,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.field_metadata import (  # noqa: E501
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.service.metadata_summary import (  # noqa: E501
    get_file_information,
    get_observation_parameters,
    get_station_configuration,
    get_timestamps,
)


def test_should_get_observation_parameters_of_one_ms():
    # Timestamps mark the centre of each 10 s integration: the observation
    # spans half an integration before the first and after the last one
    assert get_observation_parameters(
        [[10.0, 20.0, 30.0]], channels=64
    ) == ObservationParameters(
        start_time=5.0,
        observation_length=30.0,
        integration_time=10.0,
        channels=64,
    )


def test_should_span_the_timestamps_of_all_ms_in_any_order():
    parameters = get_observation_parameters(
        [[40.0, 50.0], [10.0, 20.0, 30.0]], channels=64
    )

    assert parameters.start_time == 5.0
    assert parameters.observation_length == 50.0


def test_should_get_integration_time_from_the_first_ms():
    # MSs of a field share the integration time: the second is not used
    parameters = get_observation_parameters(
        [[10.0, 20.0], [100.0, 200.0]], channels=64
    )

    assert parameters.integration_time == 10.0


def test_should_get_median_spacing_as_integration_time():
    # A gap of missing integrations does not change the integration time
    parameters = get_observation_parameters(
        [[10.0, 20.0, 30.0, 70.0, 80.0]], channels=64
    )

    assert parameters.integration_time == 10.0


def test_should_leave_channels_unset_if_unknown():
    parameters = get_observation_parameters([[10.0, 20.0]], channels=None)

    assert parameters.channels is None


def test_should_get_plain_floats():
    # The parameters are rendered and serialised: no numpy scalars
    parameters = get_observation_parameters([[10.0, 20.0]], channels=64)

    for value in (
        parameters.start_time,
        parameters.observation_length,
        parameters.integration_time,
    ):
        assert type(value) is float  # pylint: disable=unidiomatic-typecheck


def test_should_get_no_integration_time_from_a_single_timestamp():
    # Documents the current behaviour: one timestamp has no spacing, so
    # the integration time is NaN (numpy warns on an empty median)
    with pytest.warns(RuntimeWarning):
        parameters = get_observation_parameters([[10.0]], channels=64)

    assert np.isnan(parameters.integration_time)


def _metadata_summary(metadata_summary):
    return FieldMetadata(
        "field_a", "/qa/dir", sections={"metadata_summary": metadata_summary}
    )


@pytest.mark.parametrize("metadata_summary", [{}, {"stations": {}}])
def test_should_raise_if_no_stations_are_recorded(metadata_summary):
    with pytest.raises(
        RunMetadataError, match="No stations recorded for field field_a"
    ):
        get_station_configuration(_metadata_summary(metadata_summary))


@pytest.mark.parametrize("metadata_summary", [{}, {"timestamps": []}])
def test_should_raise_if_no_timestamps_are_recorded(metadata_summary):
    with pytest.raises(
        RunMetadataError, match="No timestamps recorded for field field_a"
    ):
        get_timestamps(_metadata_summary(metadata_summary))


@pytest.mark.parametrize("metadata_summary", [{}, {"MS": []}])
def test_should_raise_if_no_ms_are_recorded(metadata_summary):
    with pytest.raises(
        RunMetadataError, match="QA metadata missing for field field_a"
    ):
        get_file_information(_metadata_summary(metadata_summary))
