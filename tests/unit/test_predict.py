#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Beam tests for the ska-sdp-instrumental-calibration module."""

# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
import xarray as xr
from astropy import constants as const
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from mock import MagicMock, call, patch
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    dft_skycomponent_local,
    gaussian_tapers,
    predict_from_components,
)


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


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict.GenericBeams"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.predict.dft_skycomponent_visibility"
)
@patch("ska_sdp_instrumental_calibration.processing_tasks.predict.AltAz")
def test_predict_from_components(
    mock_altaz, mock_dft, mock_generic_beams, generate_vis
):
    vis, _ = generate_vis
    skycomp_1 = SkyComponent(
        direction=SkyCoord(ra=-0.1, dec=-26.0, unit="deg"),
        frequency=vis.frequency.data,
        name="test-comp",
        flux=np.random.rand(len(vis.frequency), 4),
        polarisation_frame=PolarisationFrame("linear"),
        shape="GAUSSIAN",
        params={"bmaj": 2 / 60.0, "bmin": 1 / 60.0, "bpa": 34},
    )
    skycomp_2 = SkyComponent(
        direction=SkyCoord(ra=-0.2, dec=-25.0, unit="deg"),
        frequency=vis.frequency.data,
        name="test-comp",
        flux=np.random.rand(len(vis.frequency), 4),
        polarisation_frame=PolarisationFrame("linear"),
        shape="GAUSSIAN",
        params={"bmaj": 2 / 60.0, "bmin": 1 / 60.0, "bpa": 34},
    )
    skycomponents = [skycomp_1, skycomp_2]

    mock_altaz.return_value = AltAz(
        obstime=Time(vis.datetime.data[0]), location=vis.configuration.location
    )

    mock_beam_instance = MagicMock()
    mock_beam_instance.array_response.return_value = np.ones((210, 4, 2, 2))
    mock_generic_beams.return_value = mock_beam_instance

    mock_transform_result = MagicMock()
    mock_transform_result.alt.degree = 45.0  # Above horizon
    mock_beam_instance.beam_direction.transform_to.return_value = (
        mock_transform_result
    )

    predict_from_components(
        vis,
        skycomponents,
        eb_coeffs="/path/to/eb",
        eb_ms="/path/to/ms",
        beam_type="everybeam",
    )

    mock_beam_instance.array_response.assert_has_calls(
        [
            call(
                direction=skycomp_1.direction,
                frequency=vis.frequency.data,
                time=np.mean(Time(vis.datetime.data)),
            ),
            call(
                direction=skycomp_2.direction,
                frequency=vis.frequency.data,
                time=np.mean(Time(vis.datetime.data)),
            ),
        ]
    )

    mock_dft.assert_has_calls(
        [
            call(vis, skycomp_1),
            call(vis, skycomp_2),
        ]
    )
