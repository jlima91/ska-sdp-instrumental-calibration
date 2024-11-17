#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam tests for the ska-sdp-instrumental-calibration module."""

# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
import xarray as xr
from astropy import constants as const
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    dft_skycomponent_local,
    gaussian_tapers,
)
from tests.test_utils import generate_vis


def test_dft_skycomponent(generate_vis):
    """Test point-source component DFT."""
    vis, _ = generate_vis
    skycomp = SkyComponent(
        direction=SkyCoord(ra=-0.1, dec=-26.0, unit="deg"),
        frequency=vis.frequency.data,
        name="test-comp",
        flux=np.random.rand(len(vis.frequency), 4),
        polarisation_frame=PolarisationFrame("linear"),
        shape="POINT",
        params={},
    )
    compvis1 = vis.assign({"vis": xr.zeros_like(vis.vis)})
    dft_skycomponent_visibility(compvis1, skycomp)
    compvis2 = vis.assign({"vis": xr.zeros_like(vis.vis)})
    dft_skycomponent_local(compvis2, skycomp)
    assert np.allclose(compvis1.vis.data, compvis2.vis.data)


def test_gaussian_tapers(generate_vis):
    """Test Gaussian component DFT."""
    vis, _ = generate_vis
    params = {
        "bmaj": 2 / 60.0,
        "bmin": 1 / 60.0,
        "bpa": 34,
    }
    skycomp = SkyComponent(
        direction=SkyCoord(ra=-0.1, dec=-26.0, unit="deg"),
        frequency=vis.frequency.data,
        name="test-comp",
        flux=np.random.rand(len(vis.frequency), 4),
        polarisation_frame=PolarisationFrame("linear"),
        shape="GAUSSIAN",
        params=params,
    )
    compvis1 = vis.assign({"vis": xr.zeros_like(vis.vis)})
    dft_skycomponent_visibility(compvis1, skycomp)
    # The ska-sdp-func version does not yet taper Gaussians.
    # Apply the gaussian_tapers taper and make does the right thing.
    uvw = np.einsum(
        "tbd,f->tbfd",
        compvis1.uvw.data,
        compvis1.frequency.data / const.c.value,  # pylint: disable=no-member
    )
    compvis1.vis.data *= gaussian_tapers(uvw, params)[..., np.newaxis]

    compvis2 = vis.assign({"vis": xr.zeros_like(vis.vis)})
    dft_skycomponent_local(compvis2, skycomp)
    assert np.allclose(compvis1.vis.data, compvis2.vis.data)
