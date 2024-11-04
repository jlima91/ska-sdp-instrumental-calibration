#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the ska_python_skeleton module."""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_create import create_visibility

from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)


@pytest.fixture
def generate_datasets():
    """Fixture to build Visibility and GainTable datasets."""
    # Create the Visibility dataset
    vis = create_visibility(
        config=create_named_configuration("LOWBD2-CORE"),
        times=[0],
        frequency=[150e6],
        phasecentre=SkyCoord(ra=0, dec=-27, unit="degree"),
        channel_bandwidth=[1e6],
        polarisation_frame=PolarisationFrame("linear"),
        weight=1.0,
    )
    # Put a point source at phase centre
    vis.vis.data[..., 0] = vis.vis.data[..., 3] = 1 + 0j

    # Create the GainTable dataset
    jones = create_gaintable_from_visibility(vis, jones_type="B")
    jones.gain.data[..., 0, 0] = 1 - 0.1j
    jones.gain.data[..., 1, 1] = 3 + 0j
    jones.gain.data += np.random.normal(0, 0.2, jones.gain.shape)
    jones.gain.data += np.random.normal(0, 0.2, jones.gain.shape) * 1j

    # Apply the GainTable dataset to the Visibility dataset
    assert np.all(vis.vis.data[..., :] == [1, 0, 0, 1])
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)
    assert np.all(vis.vis.data[..., :] != [1, 0, 0, 1])

    return vis, jones


def test_ska_python_skeleton(generate_datasets):
    """Example: Assert fixture return value."""
    vis, jones = generate_datasets
    assert np.all(vis.vis.data[..., :] != [1, 0, 0, 1])
    vis = apply_gaintable(vis=vis, gt=jones, inverse=True)
    assert np.allclose(vis.vis.data[..., :], [1, 0, 0, 1])
