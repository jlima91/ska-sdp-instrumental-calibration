import numpy as np

from ska_sdp_instrumental_calibration.data_managers.sky_model import Component


def test_deconvolve_gaussian():
    """
    Given a component, deconvolve MWA synthesised beam.
    """
    component = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=4.0,
        ref_freq=200,
        spec_idx=2.0,
        major_ax=250,
        minor_ax=200,
        pos_ang=-4,
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
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=4.0,
        ref_freq=200,
        spec_idx=2.0,
        major_ax=250,
        minor_ax=250,
        pos_ang=-4,
        beam_major=beam_major,
        beam_minor=beam_minor,
        beam_pa=beam_pa,
    )

    actual_params = np.array(component.deconvolve_gaussian())
    expectd_params = np.array((beam_major, beam_minor, 90 + beam_pa))

    np.testing.assert_allclose(expectd_params, actual_params)


def test_deconvolve_gaussian_if_major_minor_axes_are_none():
    component = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=4.0,
        ref_freq=200,
        spec_idx=2.0,
        major_ax=None,
        minor_ax=None,
        pos_ang=None,
    )

    actual_params = np.array(component.deconvolve_gaussian())
    expectd_params = np.array((0.0, 0.0, 90.0))

    np.testing.assert_allclose(expectd_params, actual_params)
