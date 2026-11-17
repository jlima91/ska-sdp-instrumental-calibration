import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from mock import ANY, Mock, patch

from ska_sdp_instrumental_calibration.xarray_processors.rotation_measures import (  # noqa: E501
    FARADAY_RM_PER_TESLA_TECU,
    RotationMeasureData,
    _phasecentre_enu,
    compute_rm_parameters,
    fit_curve,
    get_plot_params_for_station,
    get_rm_spec,
    get_stn_masks,
    model_rotations,
    model_rotations_ufunc,
    rm_from_external_tec,
    update_jones_with_masks,
)


def setup_test_data():
    nstations = 5
    nfreq = 10
    refant = 0

    gain_data = np.random.rand(
        1, nstations, nfreq, 2, 2
    ) + 1j * np.random.rand(1, nstations, nfreq, 2, 2)

    gain_data[:, :, :, 0, 0] += 0.5
    gain_data[:, :, :, 1, 1] += 0.5
    gain_data = gain_data.reshape(1, nstations, nfreq, 4)

    antenna_coords = [f"{i}" for i in range(nstations)]
    freq_coords = np.linspace(1e8, 2e8, nfreq)

    mock_gaintable = xr.Dataset(
        {
            "CALPARAM_GAIN": (
                (
                    "time",
                    "antenna_name",
                    "frequency",
                    "polarization",
                ),
                gain_data,
            ),
            "CALPARAM_WEIGHT": (
                (
                    "time",
                    "antenna_name",
                    "frequency",
                    "polarization",
                ),
                np.ones((1, nstations, nfreq, 4)),
            ),
        },
        coords={
            "time": [0],
            "antenna_name": antenna_coords,
            "frequency": freq_coords,
            "polarization": ["XX", "XY", "YX", "YY"],
        },
    )
    return nstations, nfreq, refant, mock_gaintable


def test_model_rotation_data_initialization():
    nstations, nfreq, refant, mock_gaintable = setup_test_data()

    rot_data = model_rotations(mock_gaintable, refant=refant)

    assert len(rot_data.antenna_name) == nstations
    assert len(rot_data.frequency) == nfreq

    expected_lambda_sq = [
        8.98755179,
        7.27991695,
        6.01646029,
        5.05549788,
        4.30764316,
        3.71424334,
        3.23551864,
        2.84371756,
        2.5190024,
        2.24688795,
    ]

    np.testing.assert_allclose(rot_data.lambda_sq, expected_lambda_sq)

    assert rot_data.J.shape == (1, nstations, nfreq, 4)
    assert rot_data.rm_est.shape == (
        1,
        nstations,
    )
    assert rot_data.rm_peak.shape == (
        1,
        nstations,
    )
    assert rot_data.const_rot.shape == (
        1,
        nstations,
    )
    assert rot_data.rm_spec is not None


def test_should_generate_model_rotation_data_without_refinement():
    _, __, ___, mock_gaintable = setup_test_data()

    rot_data = model_rotations(mock_gaintable, refine_fit=False)

    np.testing.assert_allclose(rot_data.const_rot, 0)


def test_should_generate_rm_spec():
    resolutions = np.array([0, 1, 1])

    l_sq = np.array([0, 1, 0])
    phi_raw = np.array([0, 100, 1000])
    mask = np.array([True, False, False])

    expected = np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j])

    out = get_rm_spec(phi_raw, mask, resolutions, l_sq, chunk_size=3)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_should_update_jones_with_mask():
    nstations = 2
    nfreq = 2

    jones = np.ones((nstations, nfreq, 2, 2), dtype=complex)
    mask = np.array([[True, False], [True, True]])
    norms = np.full((nstations, nfreq, 2, 2), 2.0)

    jones_expected = [
        [
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            ],
            [
                [
                    1.0 + 0.0j,
                    1.0 + 0.0j,
                ],
                [1.0 + 0.0j, 1.0 + 0.0j],
            ],
        ],
        [
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            ],
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            ],
        ],
    ]

    out = update_jones_with_masks(  # pylint: disable=no-member
        jones, mask, norms
    )  # pylint: disable=no-member
    np.testing.assert_allclose(out, jones_expected)


def test_get_station_masks_when_refant_weights_are_all_zeros():
    ntime, nstn, nfreq, npol1, npol2 = (1, 3, 2, 2, 2)
    weight = np.zeros((ntime, nstn, nfreq, npol1, npol2))
    weight[0, :, :, 0, 0] = 1
    weight[0, :, :, 1, 1] = 1
    weight[0, 1, :, 0, 0] = 1
    weight[0, 1, :, 1, 1] = 1
    refant = 1

    out = get_stn_masks(weight[0, 0, ...], weight[0, refant, ...])
    np.testing.assert_array_equal(out, [True, True])


def _rotation_gains(rm, frequency, const_rot=0.0):
    """Jones matrices of a Faraday rotation plus a constant rotation."""
    lambda_sq = (299792458.0 / frequency) ** 2
    phi = rm * lambda_sq + const_rot
    return np.stack(
        [
            np.stack([np.cos(phi), -np.sin(phi)], axis=-1),
            np.stack([np.sin(phi), np.cos(phi)], axis=-1),
        ],
        axis=-2,
    ).astype(complex)


def _identity_gains(nfreq):
    return np.broadcast_to(np.eye(2, dtype=complex), (nfreq, 2, 2)).copy()


def test_should_fit_curve():
    lambda_sq = np.linspace(6.0, 36.0, 50)
    true_rm, true_const = 0.3, 0.4
    # Angles are only known modulo pi.
    phi = 0.5 * np.angle(np.exp(2j * (true_rm * lambda_sq + true_const)))

    rm_est, const_rot = fit_curve(lambda_sq, phi, 0.29, rm_step=0.05)

    np.testing.assert_allclose(rm_est, true_rm, atol=1e-8)
    np.testing.assert_allclose(const_rot, true_const, atol=1e-7)


def test_should_fit_curve_using_only_unmasked_channels():
    lambda_sq = np.linspace(6.0, 36.0, 50)
    phi = 0.5 * np.angle(np.exp(2j * 0.3 * lambda_sq))
    mask = np.ones(50, dtype=bool)
    mask[::5] = False
    phi[::5] = 1.0  # garbage in masked channels
    phi[10] = np.nan

    rm_est, const_rot = fit_curve(
        lambda_sq, phi, 0.29, mask=mask, rm_step=0.05
    )

    np.testing.assert_allclose(rm_est, 0.3, atol=1e-8)
    np.testing.assert_allclose(const_rot, 0.0, atol=1e-7)


def test_should_return_initial_estimate_when_all_channels_are_masked():
    lambda_sq = np.linspace(6.0, 36.0, 5)

    out = fit_curve(lambda_sq, np.zeros(5), 0.7, mask=np.zeros(5, dtype=bool))

    assert out == (0.7, 0.0)


@pytest.mark.parametrize("descending", [False, True])
def test_should_estimate_rm_for_either_channel_order(descending):
    frequency = np.linspace(50e6, 125e6, 128)
    if descending:
        frequency = frequency[::-1]
    weight = np.ones((128, 2, 2))

    _, _, rm_est, rm_peak, const_rot = model_rotations_ufunc(
        gain=_rotation_gains(0.3, frequency, const_rot=0.2),
        weight=weight,
        gain_refant=_identity_gains(128),
        weight_refant=weight,
        frequency=frequency,
    )

    np.testing.assert_allclose(rm_est, 0.3, atol=1e-6)
    np.testing.assert_allclose(const_rot, 0.2, atol=1e-5)
    assert abs(rm_peak - 0.3) < 0.02


def test_should_ignore_flagged_and_non_finite_channels():
    frequency = np.linspace(50e6, 125e6, 128)
    gain = _rotation_gains(0.3, frequency)
    weight = np.ones((128, 2, 2))
    gain[::7] = 0  # flagged channels holding zero gains
    weight[::7] = 0
    gain[40] = np.nan  # NaN in an unflagged channel

    J, rm_spec, rm_est, _, const_rot = model_rotations_ufunc(
        gain=gain,
        weight=weight,
        gain_refant=_identity_gains(128),
        weight_refant=np.ones((128, 2, 2)),
        frequency=frequency,
    )

    assert np.all(np.isfinite(J))
    assert np.all(np.isfinite(rm_spec))
    np.testing.assert_allclose(rm_est, 0.3, atol=1e-6)
    np.testing.assert_allclose(const_rot, 0.0, atol=1e-5)


def test_should_not_refine_when_no_rotation_is_detected():
    rng = np.random.default_rng(0)
    frequency = np.linspace(50e6, 125e6, 128)
    noise = rng.normal(size=(128, 2, 2)) + 1j * rng.normal(size=(128, 2, 2))
    weight = np.ones((128, 2, 2))

    _, rm_spec, rm_est, rm_peak, const_rot = model_rotations_ufunc(
        gain=noise,
        weight=weight,
        gain_refant=_identity_gains(128),
        weight_refant=weight,
        frequency=frequency,
        peak_threshold=0.9,
    )

    assert np.max(np.abs(rm_spec)) < 0.9
    assert (rm_est, rm_peak, const_rot) == (0.0, 0.0, 0.0)


def test_should_return_zero_rotation_for_reference_and_flagged_station():
    frequency = np.linspace(50e6, 125e6, 64)
    gain = _rotation_gains(0.3, frequency)
    weight = np.ones((64, 2, 2))

    reference = model_rotations_ufunc(
        gain=gain.copy(),
        weight=weight,
        gain_refant=gain.copy(),
        weight_refant=weight,
        frequency=frequency,
    )
    flagged = model_rotations_ufunc(
        gain=gain.copy(),
        weight=np.zeros((64, 2, 2)),
        gain_refant=_identity_gains(64),
        weight_refant=weight,
        frequency=frequency,
    )

    assert reference[2:] == (0.0, 0.0, 0.0)
    assert flagged[2:] == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(flagged[1], 0)


def test_should_limit_rm_grid_to_max_rm():
    frequency = np.arange(100e6, 110e6, 10e3)

    _, rm_vals_full = compute_rm_parameters(frequency, 5)
    _, rm_vals = compute_rm_parameters(frequency, 5, max_rm=20.0)

    assert rm_vals.size < rm_vals_full.size / 10
    assert 20.0 <= np.max(np.abs(rm_vals)) < 20.0 + (rm_vals[1] - rm_vals[0])


def test_should_compute_same_rm_grid_for_descending_frequencies():
    frequency = np.linspace(100e6, 200e6, 20)

    _, rm_vals = compute_rm_parameters(frequency, 2)
    _, rm_vals_descending = compute_rm_parameters(frequency[::-1], 2)

    np.testing.assert_allclose(rm_vals_descending, rm_vals)
    assert rm_vals.size > 0


def test_should_raise_for_fewer_than_two_channels():
    with pytest.raises(ValueError, match="At least two frequency channels"):
        compute_rm_parameters(np.array([1e8]), 5)


def test_should_raise_for_duplicate_channels():
    with pytest.raises(ValueError, match="must be distinct"):
        compute_rm_parameters(np.array([1e8, 1e8]), 5)


def test_should_compute_rm_parameters():
    freq = [1.0e9, 1.1e9]
    oversample = 2

    l_sq, rm_vals = compute_rm_parameters(freq, oversample)

    np.testing.assert_allclose(l_sq, [0.089876, 0.074277], rtol=10e-5)
    np.testing.assert_allclose(
        rm_vals, [-64.10984, -32.05492, 0.0, 32.054924], rtol=10e-5
    )


def test_should_get_plot_params_for_station():
    resolutions = np.array([1, 2])
    l_sq = np.array([1, 2])
    rm_data = RotationMeasureData.constructor(
        time=np.array([1, 2]),
        antenna_name=np.array([1, 2]),
        frequency=np.array([1, 2]),
        resolution=resolutions,
        polarization=["XX"],
        lambda_sq=l_sq,
        rm_spec=np.arange(8).reshape(2, 2, 2),
        rm_est=np.arange(4).reshape(2, 2),
        rm_peak=np.arange(4).reshape(2, 2),
        const_rot=np.arange(4).reshape(2, 2),
        J=np.arange(8).reshape(2, 2, 2, 1),
    )

    plot_params = get_plot_params_for_station(rm_data, 1, 0, time=None)
    assert plot_params["stn"] == 1
    assert plot_params["xlim"].compute() == 30
    assert all(plot_params["rm_vals"] == resolutions)
    assert all(plot_params["lambda_sq"] == l_sq)
    np.testing.assert_allclose(plot_params["rm_spec"], [[2, 3], [6, 7]])
    np.testing.assert_allclose(plot_params["rm_peak"], [1, 3])
    np.testing.assert_allclose(plot_params["rm_est"], [1, 3])
    np.testing.assert_allclose(plot_params["rm_est_refant"], [0, 2])


def test_should_get_plot_params_for_station_and_time():
    resolutions = np.array([1, 2])
    l_sq = np.array([1, 2])
    rm_data = RotationMeasureData.constructor(
        time=np.array([1, 2]),
        antenna_name=np.array([1, 2]),
        frequency=np.array([1, 2]),
        resolution=resolutions,
        polarization=["XX"],
        lambda_sq=l_sq,
        rm_spec=np.arange(8).reshape(2, 2, 2),
        rm_est=np.arange(4).reshape(2, 2),
        rm_peak=np.arange(4).reshape(2, 2),
        const_rot=np.arange(4).reshape(2, 2),
        J=np.arange(8).reshape(2, 2, 2, 1),
    )

    plot_params = get_plot_params_for_station(rm_data, 1, 0)
    assert plot_params["stn"] == 1
    assert plot_params["xlim"].compute() == 10
    assert all(plot_params["rm_vals"] == resolutions)
    assert all(plot_params["lambda_sq"] == l_sq)
    np.testing.assert_allclose(plot_params["rm_spec"], [2, 3])
    np.testing.assert_allclose(plot_params["rm_peak"], [1])
    np.testing.assert_allclose(plot_params["rm_est"], [1])
    np.testing.assert_allclose(plot_params["rm_est_refant"], [0])


def _rotation_data_with_rm(rm_est):
    """RotationMeasureData whose per-station values encode the station."""
    rm_est = np.asarray(rm_est, dtype=float)
    n_time, n_ant = rm_est.shape
    station = np.arange(n_ant, dtype=float)[np.newaxis, :]
    return RotationMeasureData.constructor(
        time=np.arange(n_time),
        antenna_name=np.array([f"s{i}" for i in range(n_ant)]),
        frequency=np.array([1.0, 2.0]),
        resolution=np.array([0.0, 1.0, 2.0]),
        polarization=["XX", "XY", "YX", "YY"],
        lambda_sq=np.array([1.0, 2.0]),
        rm_spec=np.broadcast_to(
            station[..., np.newaxis], (n_time, n_ant, 3)
        ).astype(complex),
        rm_est=rm_est,
        rm_peak=10 + np.broadcast_to(station, (n_time, n_ant)),
        const_rot=np.zeros((n_time, n_ant)),
        J=np.broadcast_to(
            station[..., np.newaxis, np.newaxis], (n_time, n_ant, 2, 4)
        ).astype(complex),
    )


def test_should_pick_station_with_largest_rm_relative_to_refant():
    # Relative to refant 1 (RM 0.5): |rm - 0.5| = [0.6, 0, 0.3, 0.2].
    rm_data = _rotation_data_with_rm([[-0.1, 0.5, 0.8, 0.3]])

    plot_params = get_plot_params_for_station(rm_data, None, 1)

    assert int(plot_params["stn"].compute()) == 0
    np.testing.assert_allclose(plot_params["rm_est"], -0.1)
    np.testing.assert_allclose(plot_params["rm_peak"], 10.0)
    np.testing.assert_allclose(plot_params["rm_spec"], [0, 0, 0])
    np.testing.assert_allclose(plot_params["rm_est_refant"], 0.5)
    assert plot_params["J"].shape == (2, 2, 2)


def test_should_pick_largest_rm_station_over_all_times():
    rm_data = _rotation_data_with_rm(
        [[0.0, 0.2, 0.1, 0.0], [0.0, 0.1, -0.9, 0.3]]
    )

    plot_params = get_plot_params_for_station(rm_data, None, 0, time=None)

    assert int(plot_params["stn"].compute()) == 2
    np.testing.assert_allclose(plot_params["rm_est"], [0.1, -0.9])


def test_should_keep_given_station_for_plot():
    rm_data = _rotation_data_with_rm([[-0.1, 0.5, 0.8, 0.3]])

    plot_params = get_plot_params_for_station(rm_data, 3, 1)

    assert plot_params["stn"] == 3
    np.testing.assert_allclose(plot_params["rm_est"], 0.3)


def test_should_raise_value_error_for_freq_not_provided():
    with pytest.raises(
        ValueError,
        match=r"frequency must be provided if rm_vals or lambda_sq are None",
    ):
        model_rotations_ufunc(
            None,
            None,
            None,
            None,
            rm_vals=None,
            lambda_sq=None,
            frequency=None,
        )


def test_faraday_constant_matches_itu_r_p531():
    # 236 * B * TEC / f_GHz^2 = RM * (c / f)^2
    c = 299792458.0
    np.testing.assert_allclose(FARADAY_RM_PER_TESLA_TECU, 236e18 / c**2)


def test_phasecentre_enu_points_up_for_source_at_zenith():
    location = EarthLocation.from_geodetic(116.76, -26.82, 377.0)
    time_mjd = np.array([51546.9375])
    gast = (
        Time(time_mjd, format="mjd", scale="utc")
        .sidereal_time("apparent", "greenwich")
        .rad
    )
    # Hour angle zero and declination equal to latitude: zenith.
    phasecentre = SkyCoord(
        ra=(gast[0] + location.lon.rad) % (2 * np.pi),
        dec=location.lat.rad,
        unit="rad",
    )

    enu = _phasecentre_enu(
        phasecentre,
        np.array([location.lon.rad]),
        np.array([location.lat.rad]),
        time_mjd,
    )

    assert enu.shape == (1, 1, 3)
    np.testing.assert_allclose(enu[0, 0], [0.0, 0.0, 1.0], atol=1e-12)


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "rotation_measures._phasecentre_enu"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "rotation_measures.ppigrf"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors."
    "rotation_measures.sample_external_tec"
)
def test_rm_from_external_tec_uses_field_along_propagation(
    sample_external_tec_mock, ppigrf_mock, phasecentre_enu_mock
):
    location = EarthLocation.from_geodetic([116.7, 116.8], [-26.8, -26.9])
    xyz = np.stack([c.to_value("m") for c in location.geocentric], axis=-1)
    vis = Mock(name="vis")
    vis.configuration.xyz.data = xyz
    solution_times = np.array([4.4e9, 4.4e9 + 10.0])

    tec = np.array([[0.01, 0.02], [0.03, -0.04]])
    sample_external_tec_mock.return_value = tec
    # Field (east, north, up) in nT, shape (n_dates, antenna).
    b_east = np.array([[10.0, 20.0]])
    b_north = np.array([[25000.0, 26000.0]])
    b_up = np.array([[49000.0, 48000.0]])
    ppigrf_mock.igrf.return_value = (b_east, b_north, b_up)
    # Source at zenith for both stations and times.
    phasecentre_enu_mock.return_value = np.broadcast_to(
        [0.0, 0.0, 1.0], (2, 2, 3)
    )

    rm = rm_from_external_tec(vis, "screen.fits", solution_times)

    sample_external_tec_mock.assert_called_once_with(vis, "screen.fits", ANY)
    np.testing.assert_allclose(
        sample_external_tec_mock.call_args.args[2], solution_times
    )
    # Propagation is downwards, so B_parallel = -B_up. The pipeline's sky
    # frame rotates opposite to OSKAR's beam basis, hence the outer minus.
    expected = -FARADAY_RM_PER_TESLA_TECU * (-b_up * 1e-9) * tec
    np.testing.assert_allclose(rm, expected)
