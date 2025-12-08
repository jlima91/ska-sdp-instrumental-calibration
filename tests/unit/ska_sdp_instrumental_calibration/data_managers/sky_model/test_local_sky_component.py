import numpy as np
from astropy.coordinates import SkyCoord
from mock import Mock
from ska_sdp_datamodels.science_data_model import PolarisationFrame

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    Component,
    LocalSkyComponent,
)


def test_create_model_to_skycomponents():
    component = Component(
        name="J12345",
        RAdeg=260,
        DEdeg=-85,
        flux=4.0,
        ref_freq=200,
        alpha=2.0,
    )

    component.deconvolve_gaussian = Mock(
        name="deconvolve_gaussian", return_value=(7200, 9000, 5.0)
    )

    actual_component = LocalSkyComponent.create_from_component(
        component, [400, 800]
    )

    component.deconvolve_gaussian.assert_called_once()

    assert actual_component.direction == SkyCoord(ra=260, dec=-85, unit="deg")
    assert actual_component.name == "J12345"
    assert actual_component.polarisation_frame == PolarisationFrame("linear")
    assert actual_component.shape == "GAUSSIAN"
    assert actual_component.params == {
        "bmaj": 2.0,
        "bmin": 2.5,
        "bpa": 5.0,
    }
    np.testing.assert_allclose(
        actual_component.frequency, np.array([400, 800])
    )
    np.testing.assert_allclose(
        actual_component.flux, np.array([[16, 0, 0, 16], [64, 0, 0, 64]])
    )


def test_create_point_source_to_skycomponent():
    component = Component(
        name="J12345",
        RAdeg=260,
        DEdeg=-85,
        flux=4.0,
        ref_freq=200,
        alpha=2.0,
    )

    component.deconvolve_gaussian = Mock(
        name="deconvolve_gaussian", return_value=(0, 0, 0)
    )

    actual_component = LocalSkyComponent.create_from_component(
        component, [400, 800]
    )

    component.deconvolve_gaussian.assert_called_once()

    assert actual_component.shape == "POINT"
    assert actual_component.params == {}
