from io import StringIO

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from mock import MagicMock, patch
from ska_sdp_datamodels.science_data_model import PolarisationFrame

from ska_sdp_instrumental_calibration.numpy_processors.lsm import (
    Component,
    convert_model_to_skycomponents,
    deconvolve_gaussian,
    generate_lsm_from_csv,
    generate_lsm_from_gleamegc,
)

GLEAMFILE_LINE = """
GLEAM J235139-894114 -0.001036 0.026022 23 51 39.45 -89 41 14.30 357.914368   0.002407 -89.687309   0.002410   0.262282 0.026270   0.248581  2.95339e-02  2.19263e+02  1.11849e+01 146.4811    5.638908  -4.158033    0.446378 -0.011716 0.015203 80 3 221.397 152.189  22.430508  0.116048 0.230260    0.531743 0.199787    0.528997 0.198755  5.18718e+02 443.560  -4.158033  0.150878  0.245333 519.623 445.444  14.269532  0.064158 0.170919    0.512256  0.152039    0.509285  0.151157  4.68570e+02 376.896  -4.158033  0.031757 0.134961 469.572 379.126   4.874937 -0.001032 0.226090    0.427771  0.209262    0.423489  0.207168  4.40963e+02 342.630  -4.158033 -0.011465 0.171556 442.027 345.088  13.513020  0.035764 0.245972    0.444216  0.222199    0.438262  0.219220  4.00925e+02 283.699  -4.158033 -0.031922  0.304453 402.096 286.674   2.319414  0.000086 0.097613    0.332228  0.087342    0.326553  0.085850  3.66049e+02 271.377  -4.158033  0.016291 0.086581 367.331 274.487   9.087766  0.007765 0.067661    0.296139  0.059777    0.291348  0.058810  3.32573e+02 256.153  -4.158033  0.034249 0.100245 333.983 259.448   9.828724  0.022250 0.069156    0.202653  0.061831    0.198947  0.060700  3.16835e+02 244.937  -4.158033  0.010231 0.068757 318.315 248.382   2.127662  0.012362 0.052444    0.298724  0.046800    0.292667  0.045851  3.02262e+02 230.963  -4.158033 -0.013592 0.076489 303.812 234.615  22.214157  0.008152 0.052375    0.341529  0.048493    0.331788  0.047110  2.67819e+02 183.996  -4.158033  0.006087 0.070793 269.568 188.566  20.953054  0.001960 0.050528    0.276829  0.045727    0.267163  0.044130  2.54018e+02 171.133  -4.158033  0.012933 0.094522 255.862 176.038  22.808126  0.024398 0.071617    0.248030  0.064988    0.239452  0.062741  2.46127e+02 170.159  -4.158033  0.029935 0.116816 248.029 175.092  12.885068  0.007915 0.056775    0.278406  0.052344    0.268323  0.050448  2.28098e+02 165.932  -4.158033 -0.003216 0.038754 230.149 170.987  19.574308  0.008273 0.050534    0.345853  0.045476    0.330670  0.043480  2.27678e+02 148.799  -4.158033 -0.000142 0.058509 229.733 154.417  21.433491  0.007071 0.072735    0.317384  0.065278    0.303373  0.062396  2.16998e+02 141.9525  -4.158033 -0.003245 0.054701 219.153 147.832  10.051599  0.001770 0.058237    0.320320  0.052301    0.304456  0.049711  2.13273e+02 138.8363  -4.158033 -0.005387 0.040263 215.466 144.842   6.488752 -0.004338 0.059530    0.179936  0.053834    0.168986  0.050558  2.07908e+02 130.4126  -4.158033 -0.015960 0.085525 210.156 136.790  12.743345  0.004307 0.070833    0.205871  0.064479    0.194392  0.060883  1.88324e+02 134.0238  -4.158033 -0.069456 0.078095 190.803 140.237  14.509152 -0.007659 0.112504    0.495948  0.101910    0.464646  0.095478  1.72234e+02 125.0827  -4.158033 -0.014896 0.124724 174.942 131.7186   7.999518 -0.009265 0.082080    0.167417  0.074100    0.156235  0.069151  1.71534e+02 127.1215  -4.158033 -0.038894 0.098901 174.253 133.6561  17.220127 -0.008993 0.063705    0.097147  0.058071    0.090779  0.054264  1.70630e+02 123.0671  -4.158033  0.010692 0.061223 173.363 129.8062  18.959024 -0.370882  0.206630   1.327554   0.271901  0.282624
""".strip()  # noqa: E501


# noqa: E501
CSV_CONTENT = """# Number of sources: 1434
# RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2), FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)
 357.914368, -89.687309, 2.71901e-01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+08,-3.70882e-01, 0.000000, 2.19263e+02, 1.464811e+02, -4.158033e+00
"""  # noqa: E501


@patch("builtins.open")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_generate_lsm_from_gleam_catalogue_file(
    logger_mock, path_mock, open_mock
):
    """
    Test that a if a gleam catalogue file is passed to the function
    generate_lsm_from_gleamegc, it reads the catalogue file line by line,
    and returns a list of Components, where each component satisfies
    following criterions:

    1. The integrated flux of the component in the wide range is greater
    than minimum flux density

    2. The component is within the field of view
    """
    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True

    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    fileobj = MagicMock(name="gleamfile object")
    fileobj.__iter__.return_value = [GLEAMFILE_LINE]

    open_mock.return_value = open_mock
    open_mock.__enter__.return_value = fileobj

    lsm = generate_lsm_from_gleamegc("gleamfile.dat", phasecentre, fov=10)

    open_mock.assert_called_once_with("gleamfile.dat", "r")
    fileobj.close.assert_called_once()

    logger_mock.info.assert_called_once_with("extracted 1 GLEAM components")
    logger_mock.debug.assert_called_once_with(
        "alpha: mean = -0.37, median = -0.37, used = -0.78"
    )

    assert lsm == [
        Component(
            name="J235139-894114",
            RAdeg=357.914368,
            DEdeg=-89.687309,
            flux=0.271901,
            ref_freq=200000000.0,
            alpha=-0.370882,
            major=219.263,
            minor=146.4811,
            pa=-4.158033,
            beam_major=221.397,
            beam_minor=152.189,
            beam_pa=22.430508,
        )
    ]


@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_generate_unit_flux_source_at_phase_centre_if_gleamfile_not_found(
    logger_mock, path_mock
):
    """
    Test that a point source with unit flux is generated when
    the GLEAM catalogue file is not found.
    """
    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.degree = 2.0
    phasecentre.dec.degree = 5.0

    path_mock.return_value = path_mock
    path_mock.is_file.return_value = False

    lsm = generate_lsm_from_gleamegc("gleamfile.dat", phasecentre)

    path_mock.assert_called_once_with("gleamfile.dat")
    logger_mock.warning.assert_called_once_with(
        "Cannot open gleam catalogue file gleamfile.dat. "
        "Returning point source with unit flux at phase centre."
    )

    assert lsm == [Component(name="default", RAdeg=2.0, DEdeg=5.0, flux=1.0)]


@patch("builtins.open")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_should_exclude_component_when_flux_is_less_than_min_flux(
    logger_mock, path_mock, open_mock
):
    """
    Test that for a Component in the gleam catalogue file,
    if its integrated flux in the wide range is less than
    minimum flux density, then its excluded from the list of components.
    """
    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True

    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    fileobj = MagicMock(name="gleamfile object")
    fileobj.__iter__.return_value = [GLEAMFILE_LINE]

    open_mock.return_value = open_mock
    open_mock.__enter__.return_value = fileobj

    lsm = generate_lsm_from_gleamegc(
        "gleamfile.dat", phasecentre, flux_limit=1.0
    )

    open_mock.assert_called_once_with("gleamfile.dat", "r")
    fileobj.close.assert_called_once()

    logger_mock.info.assert_called_once_with("extracted 0 GLEAM components")
    logger_mock.debug.assert_called_once_with(
        "alpha: mean = nan, median = nan, used = -0.78"
    )

    assert lsm == []


@patch("builtins.open")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_should_set_flux_alpha_to_defaults_when_fitted_data_is_unspecified(
    logger_mock, path_mock, open_mock
):
    """
    Test that for a Component in the gleam catalogue file,
    if its integrated flux at 200 Mhz is unspecified, then:

    1. flux of the component is set to its integrated flux in the wide range

    2. spectral index (alpha) of the component is set to the nominal
    spectral index value (passed as a parameter)
    """
    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True

    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    fileobj = MagicMock(name="gleamfile object")
    line = GLEAMFILE_LINE[:3135] + "  ---     " + GLEAMFILE_LINE[3145:]
    fileobj.__iter__.return_value = [line]

    open_mock.return_value = open_mock
    open_mock.__enter__.return_value = fileobj

    lsm = generate_lsm_from_gleamegc(
        "gleamfile.dat", phasecentre, fov=10, alpha0=-0.65
    )

    open_mock.assert_called_once_with("gleamfile.dat", "r")
    fileobj.close.assert_called_once()

    logger_mock.info.assert_called_once_with("extracted 1 GLEAM components")
    logger_mock.debug.assert_called_once_with(
        "alpha: mean = nan, median = nan, used = -0.65"
    )

    # flux and alpha are set to Fintwide and alpha0 respectively
    assert lsm == [
        Component(
            name="J235139-894114",
            RAdeg=357.914368,
            DEdeg=-89.687309,
            flux=0.248581,
            ref_freq=200000000.0,
            alpha=-0.65,
            major=219.263,
            minor=146.4811,
            pa=-4.158033,
            beam_major=221.397,
            beam_minor=152.189,
            beam_pa=22.430508,
        )
    ]


@patch("builtins.open")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
def test_exclude_component_when_its_out_of_fov(path_mock, open_mock):
    """
    Test that for a Component in the gleam catalogue file,
    if its position is far away from the phase centre (outside of the
    field of view), then its excluded from the final list of components.
    """
    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True

    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    fileobj = MagicMock(name="gleamfile object")
    fileobj.__iter__.return_value = [GLEAMFILE_LINE]

    open_mock.return_value = open_mock
    open_mock.__enter__.return_value = fileobj

    lsm = generate_lsm_from_gleamegc("gleamfile.dat", phasecentre)

    assert lsm == []


@patch(
    "ska_sdp_instrumental_calibration.numpy_processors.lsm"
    ".deconvolve_gaussian"
)
def test_convert_model_to_skycomponents(deconvolve_gaussian_mock):
    """
    Given a list of Components and range frequencies over which
    we wish to store the flux information, for each component:
    1. perform power law scaling
    2. deconvove gausssian to get beam information
    3. create a Skycomponent (defined in ska-sdp-datamodels)

    This test uses a dummy gaussian source as compoenent.
    """
    deconvolve_gaussian_mock.return_value = (7200, 9000, 5.0)

    component = Component(
        name="J12345",
        RAdeg=260,
        DEdeg=-85,
        flux=4.0,
        ref_freq=200,
        alpha=2.0,
    )

    skycomp = convert_model_to_skycomponents([component], [400, 800])

    deconvolve_gaussian_mock.assert_called_once_with(component)

    assert len(skycomp) == 1
    # SkyComponent does not implement @dataclass or __equal__
    actual_component = skycomp[0]
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


@patch(
    "ska_sdp_instrumental_calibration.numpy_processors.lsm"
    ".deconvolve_gaussian"
)
def test_convert_point_source_to_skycomponent(deconvolve_gaussian_mock):
    """
    Given a list of Components and range frequencies over which
    we wish to store the flux information,
    if a component in the list is a point source,
    then the shape and parameters of the Skycomponent must be set
    appropriately.
    """
    deconvolve_gaussian_mock.return_value = (0, 0, 0)

    component = Component(
        name="J12345",
        RAdeg=260,
        DEdeg=-85,
        flux=4.0,
        ref_freq=200,
        alpha=2.0,
    )

    skycomp = convert_model_to_skycomponents([component], [400, 800])

    deconvolve_gaussian_mock.assert_called_once_with(component)

    assert len(skycomp) == 1
    actual_component = skycomp[0]
    assert actual_component.shape == "POINT"
    assert actual_component.params == {}


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

    actual_params = np.array(deconvolve_gaussian(component))
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

    actual_params = np.array(deconvolve_gaussian(component))
    expectd_params = np.array((beam_major, beam_minor, 90 + beam_pa))

    np.testing.assert_allclose(expectd_params, actual_params)


@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
def test_should_raise_exception_for_invalid_csv_file(path_mock):

    path_mock.return_value = path_mock
    path_mock.is_file.return_value = False

    phasecentre = MagicMock(name="phasecentre")

    with pytest.raises(ValueError):
        generate_lsm_from_csv("sky_model.csv", phasecentre, fov=10)


@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_should_generate_lsm_from_csv_file(logger_mock, path_mock):

    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True
    csv_file = StringIO(CSV_CONTENT)

    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    lsm = generate_lsm_from_csv(csv_file, phasecentre, fov=10)

    logger_mock.info.assert_called_once_with("extracted 1 csv components")

    assert lsm == [
        Component(
            name="comp0",
            RAdeg=357.914368,
            DEdeg=-89.687309,
            flux=0.271901,
            ref_freq=200000000.0,
            alpha=-0.370882,
            major=219.263,
            minor=146.4811,
            pa=-4.158033,
        )
    ]


@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_should_exclude_csv_comp_when_flux_is_less_than_min_flux(
    logger_mock, path_mock
):

    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True

    csv_file = StringIO(CSV_CONTENT)
    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    lsm = generate_lsm_from_csv(csv_file, phasecentre, fov=10, flux_limit=1.0)

    logger_mock.info.assert_called_once_with("extracted 0 csv components")

    assert lsm == []


@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.Path")
@patch("ska_sdp_instrumental_calibration.numpy_processors.lsm.logger")
def test_exclude_csv_comp_when_its_out_of_fov(logger_mock, path_mock):

    path_mock.return_value = path_mock
    path_mock.is_file.return_value = True
    csv_file = StringIO(CSV_CONTENT)

    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180

    lsm = generate_lsm_from_csv(csv_file, phasecentre)

    logger_mock.info.assert_called_once_with("extracted 0 csv components")

    assert lsm == []
