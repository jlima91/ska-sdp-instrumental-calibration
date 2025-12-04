import numpy as np

from ska_sdp_instrumental_calibration.data_managers.sky_model import Component


def test_deconvolve_gaussian():
    """
    Given a component, deconvolve MWA synthesised beam.
    """
    component = Component(
        name="J12345",
        RAdeg=260,
        DEdeg=-85,
        flux=4.0,
        ref_freq=200,
        alpha=2.0,
        major=250,
        minor=200,
        pa=-4,
        beam_major=200,
        beam_minor=150,
        beam_pa=20,
    )

    actual_params = np.array(component.deconvolve_gaussian())
    expectd_params = np.array(
        (168.6690698011579, 107.47439179828898, -29.158834703512973)
    )

    np.testing.assert_allclose(expectd_params, actual_params)


def test_deconvolve_circular_gaussian():
    """
    Given a component, deconvolve MWA synthesised beam.
    if the gaussian is circular, then handle it appropriately.
    """
    beam_major = 200
    beam_minor = 150
    beam_pa = 20
    component = Component(
        name="J12345",
        RAdeg=260,
        DEdeg=-85,
        flux=4.0,
        ref_freq=200,
        alpha=2.0,
        major=250,
        minor=250,
        pa=-4,
        beam_major=beam_major,
        beam_minor=beam_minor,
        beam_pa=beam_pa,
    )

    actual_params = np.array(component.deconvolve_gaussian())
    expectd_params = np.array((beam_major, beam_minor, 90 + beam_pa))

    np.testing.assert_allclose(expectd_params, actual_params)
