#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam tests for the ska-sdp-instrumental-calibration module."""

# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.processing_tasks.beams import (
    GenericBeams,
)
from tests.test_utils import generate_vis, oskar_ms

# eb_ms = str(untar("data/OSKAR_MOCK.ms.tar.gz"))


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.beams.eb.load_telescope"
)
def test_beam_creation_low(mock_telescope, generate_vis, oskar_ms):
    """Test Low beam model creation."""
    vis, _ = generate_vis
    mock_telescope.return_value = "mock_telescope"
    eb_ms = oskar_ms

    beams = GenericBeams(vis=vis, array="LOW", ms_path=eb_ms)
    assert beams.beam_direction == vis.phasecentre
    assert np.all(beams.antenna_names == vis.configuration.names.data)
    assert beams.array_location == vis.configuration.location
    assert beams.array == "low"
    assert beams.telescope == "mock_telescope"
    # Also test that the beam type determined correctly
    beams = GenericBeams(vis=vis, ms_path=eb_ms)
    assert beams.array == "low"


def test_beam_creation_mid(generate_vis):
    """Test Mid beam model creation."""
    vis, _ = generate_vis
    beams = GenericBeams(vis=vis, array="Mid")
    assert beams.beam_direction == vis.phasecentre
    assert np.all(beams.antenna_names == vis.configuration.names.data)
    assert beams.array_location == vis.configuration.location
    assert beams.array == "mid"


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.beams.eb.load_telescope"
)
def test_update_beam_direction_low(mock_telescope, generate_vis, oskar_ms):
    """Test the update_beam_direction function."""
    vis, _ = generate_vis
    eb_ms = oskar_ms
    mock_telescope.return_value = "mock_telescope"

    beams = GenericBeams(vis=vis, array="LOW", ms_path=eb_ms)
    assert beams.beam_direction == vis.phasecentre
    direction = SkyCoord("1h", "-30d", frame="icrs")
    beams.update_beam_direction(direction)
    assert beams.beam_direction == direction


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

    beams = GenericBeams(vis=vis, array="LOW", ms_path=eb_ms)

    # Test an unnormalised beam about a degree off centre
    #  - although the response is constant, so it doesn't matter where it is
    direction = SkyCoord("0h", "-28d", frame="icrs")
    time = Time(vis.time.data[0] / 24 / 3600, format="mjd")
    frequency = vis.frequency.data
    gain = beams.array_response(direction, frequency, time)
    assert 0 < np.max(np.abs(gain)) < 1e-2
    mock.station_response.assert_called()

    # Normalisation is only for OSKAR telescope type. Here the type is
    # MagicMock so no normalisation should occur
    beams.update_beam(frequency=frequency, time=time)
    gain2 = beams.array_response(direction, frequency, time)
    mock.station_response.assert_called()
    assert np.allclose(gain2, gain)

    # Force the normalisation
    beams.set_scale = "oskar"
    beams.update_beam(frequency=frequency, time=time)
    gain = beams.array_response(direction, frequency, time)
    mock.station_response.assert_called()
    assert np.max(np.abs(gain)) > 0.99
    assert np.abs(np.linalg.norm(gain[0, 0]) / np.sqrt(2) - 1.0) < 1e-7
    assert beams.set_scale is None


def test_array_response_mid(generate_vis):
    """Check the returned beam Jones matrices."""
    # Mid beams are not yet set, so should default to identity matrices
    vis, _ = generate_vis
    beams = GenericBeams(vis=vis, array="MID")
    direction = SkyCoord("0h", "-28d", frame="icrs")
    frequency = vis.frequency.data
    gain = beams.array_response(direction, frequency)
    assert np.allclose(gain[..., :, :], np.eye(2))


def test_beam_creation_invalid_vis_type():
    """Test that GenericBeams raises ValueError when vis is not an xr.Dataset."""
    # Create a non-xarray input
    invalid_vis = {"data": "not_an_xarray_dataset"}

    # Check that the correct error is raised with the expected message
    with pytest.raises(ValueError, match=r"vis is not of type xr\.Dataset:.*"):
        GenericBeams(vis=invalid_vis)

    # Test with another invalid type
    invalid_vis = np.array([1, 2, 3])
    with pytest.raises(ValueError, match=r"vis is not of type xr\.Dataset:.*"):
        GenericBeams(vis=invalid_vis)


def test_beam_creation_set_direction(generate_vis):
    vis, _ = generate_vis

    direction = SkyCoord("1h", "-30d", frame="icrs")
    beams = GenericBeams(vis=vis, array="MID", direction=direction)
    assert beams.beam_direction == direction


@patch("ska_sdp_instrumental_calibration.processing_tasks.beams.logger")
def test_beam_creation_warning_when_pointing_below_horizon(
    logger_mock, generate_vis
):
    vis, _ = generate_vis

    # Point to a random position below horizon
    direction = SkyCoord("12h", "70d", frame="icrs")
    beams = GenericBeams(vis=vis, array="MID", direction=direction)
    frequency = vis.frequency.data
    beams.array_response(
        direction,
        frequency,
        time=Time(vis.time.data[0] / 24 / 3600, format="mjd"),
    )
    logger_mock.warning.assert_has_calls(
        [
            call("pointing below horizon: %.f deg", -46.828392486474634),
            call(
                "The Mid beam model is not current set. Only use with compact, centred sky models."
            ),
            call("Direction below horizon. Returning zero gains."),
        ]
    )


def test_low_array_requires_ms_path(generate_vis):
    """Test that ValueError is raised if ms_path is not provided for Low array."""
    vis, _ = generate_vis
    with pytest.raises(
        ValueError, match="Low array requires ms_path for everybeam."
    ):
        GenericBeams(vis=vis, array="LOW")


@patch("ska_sdp_instrumental_calibration.processing_tasks.beams.logger")
def test_unknown_beam_type_logs_info(logger_mock, generate_vis):
    """Test that an info log is made for unknown beam types."""
    vis, _ = generate_vis
    GenericBeams(vis=vis, array="UNKNOWN")
    logger_mock.info.assert_called_with("Unknown beam")
