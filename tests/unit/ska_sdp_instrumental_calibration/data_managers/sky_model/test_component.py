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
        spec_idx=[2.0],
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
        spec_idx=[2.0],
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
        spec_idx=[2.0],
        major_ax=None,
        minor_ax=None,
        pos_ang=None,
    )

    actual_params = np.array(component.deconvolve_gaussian())
    expectd_params = np.array((0.0, 0.0, 90.0))

    np.testing.assert_allclose(expectd_params, actual_params)


def test_log_model_constant_spectral_index_vectorised():
    component = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=2.5,
        ref_freq=1.0e8,
        spec_idx=[-0.7],
    )

    freq = np.array([0.5e8, 1.0e8, 2.0e8])
    result = component.calculate_flux(freq)

    expected = component.i_pol * (freq / component.ref_freq) ** (-0.7)
    assert np.allclose(result, expected)


def test_use_logarthmic_polynomial_method_when_log_spec_id_is_true():
    flux = 1.0
    ref_freq = 1.0e8
    freq = np.array([0.5e8, 1.0e8, 2.0e8])

    # spec_idx encodes exponent(log10(ratio)) = a0 + a1*log10(ratio)
    a0, a1, a2, a3 = -0.2, 0.05, 0.6, 0.78
    spec_idx = [a0, a1, a2, a3]

    component = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=flux,
        ref_freq=ref_freq,
        spec_idx=spec_idx,
    )

    ratio = freq / ref_freq
    log_ratio = np.log10(ratio)
    exponent = a0 + a1 * log_ratio + a2 * log_ratio**2 + a3 * log_ratio**3
    expected = flux * (ratio**exponent)

    result = component.calculate_flux(freq)
    assert np.allclose(result, expected)


def test_use_linear_polynomial_method_when_log_spec_id_is_false():
    flux = 10.0
    ref_freq = 100.0
    freq = np.array([50.0, 100.0, 200.0])
    spec_idx = [-0.7]

    component = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=flux,
        ref_freq=ref_freq,
        spec_idx=spec_idx,
        log_spec_idx=False,
    )

    ratio = freq / ref_freq
    expected = flux + (-0.7) * (ratio - 1.0)

    result = component.calculate_flux(freq)
    assert np.allclose(result, expected)


def test_if_spec_idx_is_none_then_return_reference_flux_log_and_linear():
    flux = 3.3
    ref_freq = 1.0e8
    freq = np.array([0.8e8, 1.0e8, 1.2e8])

    component_false = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=flux,
        ref_freq=ref_freq,
        spec_idx=None,
        log_spec_idx=False,
    )

    component_true = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=flux,
        ref_freq=ref_freq,
        spec_idx=None,
        log_spec_idx=True,
    )

    out_log = component_true.calculate_flux(freq)
    out_lin = component_false.calculate_flux(freq)
    expected = np.full(shape=len(freq), fill_value=flux)

    assert np.allclose(out_log, expected)
    assert np.allclose(out_lin, expected)


def test_if_spec_idx_is_empty_list_returns_reference_flux_log_and_linear():
    flux = 3.3
    ref_freq = 1.0e8
    freq = np.array([0.8e8, 1.0e8, 1.2e8])

    component_false = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=flux,
        ref_freq=ref_freq,
        spec_idx=None,
        log_spec_idx=False,
    )

    component_true = Component(
        component_id="J12345",
        ra=260,
        dec=-85,
        i_pol=flux,
        ref_freq=ref_freq,
        spec_idx=None,
        log_spec_idx=True,
    )

    out_log = component_true.calculate_flux(freq)
    out_lin = component_false.calculate_flux(freq)

    expected = np.full(shape=len(freq), fill_value=flux)
    assert np.allclose(out_log, expected)
    assert np.allclose(out_lin, expected)
