#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam tests for the ska-sdp-instrumental-calibration module."""

# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ska_sdp_instrumental_calibration.processing_tasks.beams import (
    GenericBeams,
)
from tests.test_utils import generate_vis, untar

eb_ms = str(untar("data/OSKAR_MOCK.ms.tar.gz"))


def test_beam_creation_low(generate_vis):
    """Test Low beam model creation."""
    vis, _ = generate_vis
    beams = GenericBeams(vis=vis, array="LOW", ms_path=eb_ms)
    assert beams.beam_direction == vis.phasecentre
    assert np.all(beams.antenna_names == vis.configuration.names.data)
    assert beams.array_location == vis.configuration.location
    assert beams.array == "low"
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


def test_update_beam_direction_low(generate_vis):
    """Test the update_beam_direction function."""
    vis, _ = generate_vis
    beams = GenericBeams(vis=vis, array="LOW", ms_path=eb_ms)
    assert beams.beam_direction == vis.phasecentre
    direction = SkyCoord("1h", "-30d", frame="icrs")
    beams.update_beam_direction(direction)
    assert beams.beam_direction == direction


@pytest.mark.skip(reason="Need everybeam coefficient data")
def test_array_response_low(generate_vis):
    """Check the returned beam Jones matrices."""
    vis, _ = generate_vis
    # Also need something like this before array_response:
    # os.environ["EVERYBEAM_DATADIR"] = eb_coeffs
    beams = GenericBeams(vis=vis, array="LOW", ms_path=eb_ms)
    # Test an unnormalised beam about a degree off centre
    direction = SkyCoord("0h", "-28d", frame="icrs")
    time = Time(vis.time.data[0] / 24 / 3600, format="mjd")
    frequency = vis.frequency.data
    gain = beams.array_response(direction, frequency, time)
    assert 0 < np.max(np.abs(gain)) < 1e-2
    # Test a normalised beam about a degree off centre
    beams.update_beam(frequency=frequency, time=time)
    gain = beams.array_response(direction, frequency, time)
    assert 0.5 < np.max(np.abs(gain)) < 1
    # Test a normalised beam at beam centre
    direction = beams.beam_direction
    gain = beams.array_response(direction, frequency, time)
    assert np.allclose(gain[..., :, :], np.eye(2))


def test_array_response_mid(generate_vis):
    """Check the returned beam Jones matrices."""
    # Mid beams are not yet set, so should default to identity matrices
    vis, _ = generate_vis
    beams = GenericBeams(vis=vis, array="MID")
    direction = SkyCoord("0h", "-28d", frame="icrs")
    frequency = vis.frequency.data
    gain = beams.array_response(direction, frequency)
    assert np.allclose(gain[..., :, :], np.eye(2))
