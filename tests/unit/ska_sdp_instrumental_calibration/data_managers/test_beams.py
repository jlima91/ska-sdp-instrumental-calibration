#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam tests for the ska-sdp-instrumental-calibration module."""

# pylint: skip-file
# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from mock import MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.data_managers.beams import BeamsFactory

# from ska_sdp_instrumental_calibration.processing_tasks.beams import (
#     GenericBeams,
# )

# eb_ms = str(untar("data/OSKAR_MOCK.ms.tar.gz"))


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.beams.eb.load_telescope"
)
def test_beam_creation_low(mock_telescope, generate_vis, oskar_ms):
    """Test Low beam model creation."""
    vis, _ = generate_vis
    mock_telescope.return_value = "mock_telescope"
    eb_ms = oskar_ms

    beams = BeamsFactory(
        vis.configuration, vis.phasecentre, array="LOW", ms_path=eb_ms
    )
    assert beams.beam_direction == vis.phasecentre
    assert np.all(beams.antenna_names == vis.configuration.names.data)
    assert beams.array_location == vis.configuration.location
    assert beams.array == "low"
    assert beams.telescope == "mock_telescope"
    # Also test that the beam type determined correctly
    beams = BeamsFactory(vis.configuration, vis.phasecentre, ms_path=eb_ms)
    assert beams.array == "low"


@pytest.mark.skip("Function signature changed")
def test_beam_creation_mid(generate_vis):
    """Test Mid beam model creation."""
    vis, _ = generate_vis
    beams = BeamsFactory(vis.configuration, vis.phasecentre, array="Mid")
    assert beams.beam_direction == vis.phasecentre
    assert np.all(beams.antenna_names == vis.configuration.names.data)
    assert beams.array_location == vis.configuration.location
    assert beams.array == "mid"


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.beams.eb.load_telescope"
)
def test_update_beam_direction_low(mock_telescope, generate_vis, oskar_ms):
    """Test the update_beam_direction function."""
    vis, _ = generate_vis
    eb_ms = oskar_ms
    mock_telescope.return_value = "mock_telescope"

    beams = BeamsFactory(
        vis.configuration, vis.phasecentre, array="LOW", ms_path=eb_ms
    )
    assert beams.beam_direction == vis.phasecentre
    direction = SkyCoord("1h", "-30d", frame="icrs")
    beams.update_beam_direction(direction)
    assert beams.beam_direction == direction


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.beams.eb.load_telescope"
)
def test_array_response_low(mock_telescope, generate_vis, oskar_ms):
    """Check the returned beam Jones matrices."""
    vis, _ = generate_vis
    eb_ms = oskar_ms
    mock = MagicMock(name="mock_telescope")
    mock.station_response.return_value = np.array(
        [
            [0.00607739 - 0.00137632j, -0.00020882 - 0.00024169j],
            [-0.00035607 + 0.00014488j, -0.00606867 + 0.00149764j],
        ]
    )
    mock_telescope.return_value = mock

    beams = BeamsFactory(
        vis.configuration, vis.phasecentre, array="LOW", ms_path=eb_ms
    )

    # Test an unnormalised beam about a degree off centre
    #  - although the response is constant, so it doesn't matter where it is
    direction = SkyCoord("0h", "-28d", frame="icrs")
    time = Time(vis.time.data[0] / 24 / 3600, format="mjd")
    frequency = vis.frequency
    beams.update_beam(frequency, time=time)
    gain = beams.array_response(direction, frequency, time)
    assert 0 < np.max(np.abs(gain)) < 1e-2
    mock.station_response.assert_called()

    # Normalisation is only for OSKAR telescope type. Here the type is
    # MagicMock so no normalisation should occur
    beams.update_beam(frequency, time=time)
    gain2 = beams.array_response(direction, frequency, time)
    mock.station_response.assert_called()
    assert np.allclose(gain2, gain)

    # Force the normalisation
    beams.set_scale = "oskar"
    beams.update_beam(frequency, time=time)
    gain = beams.array_response(direction, frequency, time)
    mock.station_response.assert_called()
    assert np.max(np.abs(gain)) > 0.99
    assert np.abs(np.linalg.norm(gain[0, 0]) / np.sqrt(2) - 1.0) < 1e-7
    assert beams.set_scale is None


@pytest.mark.skip("Function signature changed")
def test_array_response_mid(generate_vis):
    """Check the returned beam Jones matrices."""
    # Mid beams are not yet set, so should default to identity matrices
    vis, _ = generate_vis
    beams = BeamsFactory(vis.configuration, vis.phasecentre, array="Mid")
    direction = SkyCoord("0h", "-28d", frame="icrs")
    frequency = vis.frequency
    gain = beams.array_response(direction, frequency)
    assert np.allclose(gain[..., :, :], np.eye(2))


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams.logger"
)
def test_beam_creation_warning_when_pointing_below_horizon(
    logger_mock, generate_vis
):
    vis, _ = generate_vis

    # Point to a random position below horizon
    direction = SkyCoord("12h", "70d", frame="icrs")
    beams = BeamsFactory(vis.configuration, direction, array="MID")

    frequency = vis.frequency
    beams.array_response(
        direction,
        frequency,
        time=Time(vis.time.data[0] / 24 / 3600, format="mjd"),
    )
    logger_mock.warning.assert_has_calls(
        [
            # call("pointing below horizon: %.f deg", -46.828392486474634),
            call(
                "The Mid beam model is not current set. Only use with compact, centred sky models."
            ),
            call("Direction below horizon. Returning zero gains."),
        ]
    )


@pytest.mark.skip("Function signature changed")
def test_low_array_requires_ms_path(generate_vis):
    """Test that ValueError is raised if ms_path is not provided for Low array."""
    vis, _ = generate_vis
    with pytest.raises(
        ValueError, match="Low array requires ms_path for everybeam."
    ):
        BeamsFactory(vis.configuration, vis.phasecentre, array="LOW")


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams.logger"
)
def test_unknown_beam_type_logs_info(logger_mock, generate_vis):
    """Test that an info log is made for unknown beam types."""
    vis, _ = generate_vis
    BeamsFactory(vis.configuration, vis.phasecentre, array="UNKNOWN")
    logger_mock.info.assert_called_with("Unknown beam")


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams.AltAz"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams.GenericBeams"
)
def test_should_create_beams(generic_beam_mock, AltAz_mock, generate_vis):
    vis, _ = generate_vis
    vis_mock = MagicMock(name="vis")

    vis_mock.frequency = "frequency"

    time_mock = Mock(name="astro-time")
    generic_beam_mock.return_value = generic_beam_mock
    AltAz_mock.return_value = AltAz_mock
    AltAz_mock.alt.degree = 45
    generic_beam_mock.beam_direction.transform_to.return_value = AltAz_mock

    create_beams(
        time_mock,
        vis.frequency,
        vis.configuration,
        vis.phasecentre,
        "coeffs",
        "eb_ms",
    )

    generic_beam_mock.assert_called_once_with(
        configuration=vis.configuration,
        direction=vis.phasecentre,
        array="low",
        ms_path="eb_ms",
    )

    freq = generic_beam_mock.update_beam.call_args.args[0].data
    time = generic_beam_mock.update_beam.call_args.kwargs["time"]
    np.testing.assert_equal(freq, vis.frequency.data)
    assert time == time_mock


@pytest.mark.skip("Function signature changed")
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams.AltAz"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams.GenericBeams"
)
def test_should_raise_below_horizon_error_for_create_beams(
    generic_beam_mock, AltAz_mock, generate_vis
):
    vis, _ = generate_vis

    time_mock = Mock(name="astro-time")
    generic_beam_mock.return_value = generic_beam_mock
    AltAz_mock.return_value = AltAz_mock
    AltAz_mock.alt.degree = -1
    generic_beam_mock.beam_direction.transform_to.return_value = AltAz_mock

    with pytest.raises(ValueError, match="Pointing below horizon el=-1"):
        create_beams(
            time_mock,
            vis.frequency,
            vis.configuration,
            vis.phasecentre,
            "coeffs",
            "eb_ms",
        )
