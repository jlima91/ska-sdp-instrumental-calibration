import numpy as np
import pytest
from astropy import constants as const
from astropy import units
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.sky_model.sky_model_io import (
    read_skymodel_from_lsm_csv,
)

from ska_sdp_instrumental_calibration.data_managers.sky_model.sky_model_operations import (  # noqa: E501
    deconvolve_gaussian,
    dft_skycomponent,
    direction_cosines,
    gaussian_shape,
    gaussian_tapers,
    is_above_horizon,
    skycomponent_flux,
)


def make_skycomponent(**kwargs):
    defaults = {
        "component_id": "J12345",
        "source_id": "",
        "ra_deg": 10.0,
        "dec_deg": -30.0,
        "i_pol_jy": 4.0,
        "ref_freq_hz": 200.0,
        "epoch": 0.0,
    }
    defaults.update(kwargs)
    return SkyComponent(**defaults)


def test_direction_cosines_at_phase_centre():
    phase_centre = SkyCoord(ra=10, dec=-30, unit="deg")

    np.testing.assert_allclose(
        direction_cosines(10, -30, phase_centre), (0, 0, 1), atol=1e-15
    )


def test_direction_cosines_match_skycoord_to_lmn():
    phase_centre = SkyCoord(ra=10, dec=-30, unit="deg")
    direction = SkyCoord(ra=12, dec=-28, unit="deg")
    offset = direction.transform_to(phase_centre.skyoffset_frame())
    expected = offset.cartesian.xyz.value[[1, 2, 0]]

    np.testing.assert_allclose(
        direction_cosines(12, -28, phase_centre), expected
    )


@pytest.mark.parametrize(
    "a_arcsec,b_arcsec",
    [
        (None, None),
        (None, 10.0),
        (10.0, None),
        (0, 0),
        (np.nan, np.nan),
        (np.nan, 10.0),
    ],
)
def test_gaussian_shape_is_none_for_point_source(a_arcsec, b_arcsec):
    comp = make_skycomponent(a_arcsec=a_arcsec, b_arcsec=b_arcsec)

    assert gaussian_shape(comp) is None


def test_gaussian_shape_returns_shape_parameters():
    comp = make_skycomponent(a_arcsec=7200, b_arcsec=9000, pa_deg=5.0)

    assert gaussian_shape(comp) == (7200, 9000, 5.0)


@pytest.mark.parametrize("pa_deg", [None, np.nan])
def test_gaussian_shape_defaults_missing_position_angle_to_zero(pa_deg):
    comp = make_skycomponent(a_arcsec=7200, b_arcsec=9000, pa_deg=pa_deg)

    assert gaussian_shape(comp)[2] == 0.0


@pytest.mark.parametrize("alt_deg, expected", [(30, True), (-30, False)])
def test_is_above_horizon(alt_deg, expected):
    location = EarthLocation.from_geodetic(lon=116.7, lat=-26.8, height=377)
    obstime = Time("2025-01-01T00:00:00", scale="utc")
    direction = SkyCoord(
        AltAz(
            alt=alt_deg * units.deg,
            az=45 * units.deg,
            obstime=obstime,
            location=location,
        )
    ).icrs
    skycomponent = make_skycomponent(
        ra_deg=direction.ra.degree, dec_deg=direction.dec.degree
    )

    assert is_above_horizon(skycomponent, obstime, location) is expected


def test_gaussian_tapers_is_unity_at_zero_baseline():
    np.testing.assert_allclose(gaussian_tapers(0.0, 0.0, 120, 60, 30), 1.0)


def test_gaussian_tapers_halves_at_expected_baseline_length():
    # FT of a Gaussian with FWHM theta [rad] falls to 1/2 at
    # |u| = 2 ln(2) / (pi * theta) [wavelengths]. The axes are semi-axes,
    # so FWHM = 2 * axis. At zero position angle the major axis points
    # North (along v) and the minor axis points East (along u).
    a_arcsec, b_arcsec = 120.0, 60.0
    fwhm_maj, fwhm_min = np.deg2rad(
        [2 * a_arcsec / 3600.0, 2 * b_arcsec / 3600.0]
    )
    v_half = 2 * np.log(2) / (np.pi * fwhm_maj)
    u_half = 2 * np.log(2) / (np.pi * fwhm_min)

    np.testing.assert_allclose(
        gaussian_tapers(0.0, v_half, a_arcsec, b_arcsec, 0.0), 0.5
    )
    np.testing.assert_allclose(
        gaussian_tapers(u_half, 0.0, a_arcsec, b_arcsec, 0.0), 0.5
    )


def test_gaussian_tapers_position_angle_rotates_from_north_through_east():
    u = np.array([100.0, -50.0, 300.0])
    v = np.array([20.0, 200.0, -120.0])

    # Rotating the major axis from North (v) to East (u) is equivalent to
    # rotating the baselines from East (u) to North (v).
    np.testing.assert_allclose(
        gaussian_tapers(u, v, 120, 60, 90.0),
        gaussian_tapers(-v, u, 120, 60, 0.0),
    )


def test_gaussian_tapers_matches_brute_force_fourier_transform():
    a_arcsec, b_arcsec, pa_deg = 360.0, 90.0, 30.0
    fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    sig_a = np.deg2rad(2 * a_arcsec / 3600.0) * fwhm_to_sigma
    sig_b = np.deg2rad(2 * b_arcsec / 3600.0) * fwhm_to_sigma

    grid = np.linspace(-4 * sig_a, 4 * sig_a, 401)
    l_grid, m_grid = np.meshgrid(grid, grid)
    pa_rad = np.deg2rad(pa_deg)
    # l points East and m points North; the position angle is measured
    # from North through East.
    along_a = l_grid * np.sin(pa_rad) + m_grid * np.cos(pa_rad)
    along_b = -l_grid * np.cos(pa_rad) + m_grid * np.sin(pa_rad)
    image = np.exp(-0.5 * (along_a**2 / sig_a**2 + along_b**2 / sig_b**2))
    image /= image.sum()

    for u, v in [(100.0, 20.0), (-50.0, 200.0), (300.0, -120.0)]:
        brute = np.abs(
            np.sum(image * np.exp(-2j * np.pi * (u * l_grid + v * m_grid)))
        )
        np.testing.assert_allclose(
            gaussian_tapers(u, v, a_arcsec, b_arcsec, pa_deg),
            brute,
            atol=1e-4,
        )


def test_skycomponent_flux_in_linear_frame():
    comp = make_skycomponent(spec_idx=[2.0], log_spec_idx=True)

    flux = skycomponent_flux(
        comp, np.array([400, 800]), PolarisationFrame("linear")
    )

    np.testing.assert_allclose(
        flux, np.array([[16, 0, 0, 16], [64, 0, 0, 64]])
    )


def test_skycomponent_flux_in_stokes_i_frame():
    comp = make_skycomponent(spec_idx=[2.0], log_spec_idx=True)

    flux = skycomponent_flux(
        comp, np.array([400, 800]), PolarisationFrame("stokesI")
    )

    np.testing.assert_allclose(flux, np.array([[16], [64]]))


def test_dft_of_point_source_at_phase_centre_is_flux():
    comp = make_skycomponent()
    uvw = np.random.default_rng(0).uniform(-1000, 1000, (3, 6, 3))

    vis = dft_skycomponent(
        uvw, comp, SkyCoord(ra=10, dec=-30, unit="deg"), np.array([1e8, 2e8])
    )

    assert vis.shape == (3, 6, 2, 4)
    np.testing.assert_allclose(vis[..., 0], 4.0)
    np.testing.assert_allclose(vis[..., 1], 0.0)
    np.testing.assert_allclose(vis[..., 2], 0.0)
    np.testing.assert_allclose(vis[..., 3], 4.0)


def test_dft_of_offset_point_source_has_expected_phase():
    comp = make_skycomponent(ra_deg=12, dec_deg=-28)
    phase_centre = SkyCoord(ra=10, dec=-30, unit="deg")
    frequency = np.array([1e8, 2e8])
    uvw = np.random.default_rng(0).uniform(-1000, 1000, (3, 6, 3))

    vis = dft_skycomponent(
        uvw, comp, phase_centre, frequency, PolarisationFrame("stokesI")
    )

    l_comp, m_comp, n_comp = direction_cosines(12, -28, phase_centre)
    uvw_lambda = (
        uvw[:, :, np.newaxis, :]
        * frequency[:, np.newaxis]
        / const.c.value  # pylint: disable=no-member
    )
    expected = 4.0 * np.exp(
        -2j * np.pi * (uvw_lambda @ np.array([l_comp, m_comp, n_comp - 1]))
    )

    assert vis.shape == (3, 6, 2, 1)
    np.testing.assert_allclose(vis[..., 0], expected)


def test_dft_of_gaussian_source_is_tapered():
    phase_centre = SkyCoord(ra=10, dec=-30, unit="deg")
    frequency = np.array([1e8, 2e8])
    uvw = np.random.default_rng(0).uniform(-1000, 1000, (3, 6, 3))

    point = dft_skycomponent(uvw, make_skycomponent(), phase_centre, frequency)
    gaussian = dft_skycomponent(
        uvw,
        make_skycomponent(a_arcsec=120, b_arcsec=60, pa_deg=30),
        phase_centre,
        frequency,
    )

    taper = np.abs(gaussian[..., 0]) / np.abs(point[..., 0])
    assert np.all(taper < 1.0)
    assert np.all(taper > 0.0)


def test_dft_of_point_source_read_from_csv_with_empty_shape(tmp_path):
    csv_path = tmp_path / "lsm.csv"
    csv_path.write_text(
        "# (component_id,source_id,ra_deg,dec_deg,i_pol_jy,ref_freq_hz,"
        "epoch,a_arcsec,b_arcsec,pa_deg,spec_idx,log_spec_idx) = format\n"
        "# VERSION=1.0.0\n"
        "# CATALOGUE_NAME=test\n"
        "# UPLOADED_AT=2026-01-01\n"
        'J1,S1,10.0,-30.0,4.0,200,0.0,,,,"[]",\n'
    )
    comp = read_skymodel_from_lsm_csv(csv_path).components["J1"]
    uvw = np.random.default_rng(0).uniform(-1000, 1000, (3, 6, 3))

    vis = dft_skycomponent(
        uvw, comp, SkyCoord(ra=10, dec=-30, unit="deg"), np.array([1e8, 2e8])
    )

    np.testing.assert_allclose(vis[..., 0], 4.0)
    np.testing.assert_allclose(vis[..., 3], 4.0)


class TestDeconvolveGaussian:
    def test_general_case(self):
        np.testing.assert_allclose(
            deconvolve_gaussian(250, 200, -4, 200, 150, 20),
            (168.6690698011579, 107.47439179828898, -29.158834703512973),
        )

    def test_circular_case(self):
        np.testing.assert_allclose(
            deconvolve_gaussian(250, 250, -4, 200, 150, 20),
            (200, 150, 110),
        )

    def test_circular_source_smaller_than_beam_major_axis(self):
        np.testing.assert_allclose(
            deconvolve_gaussian(130, 130, 0, 135, 125, 10),
            (np.sqrt(130**2 - 125**2), 0, 100),
        )

    def test_unresolved_source_is_point_source(self):
        major, minor, _ = deconvolve_gaussian(100, 80, 0, 200, 150, 20)

        assert major == 0
        assert minor == 0

    def test_source_minor_axis_larger_than_major_is_circular(self):
        np.testing.assert_allclose(
            deconvolve_gaussian(250, 300, -4, 200, 150, 20),
            deconvolve_gaussian(250, 250, -4, 200, 150, 20),
        )

    @pytest.mark.parametrize("beam_minor", [200, 250])
    def test_beam_minor_axis_not_smaller_than_major_is_circular(
        self, beam_minor
    ):
        np.testing.assert_allclose(
            deconvolve_gaussian(250, 200, -4, 200, beam_minor, 20),
            deconvolve_gaussian(250, 200, -4, 200, 200, 20),
        )
