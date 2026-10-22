from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from mock import MagicMock, patch
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    export_lsm_to_csv,
    read_gleamegc,
    sky_model_reader,
)

ComponentConverters = sky_model_reader.ComponentConverters


GLEAMFILE_LINE = """
GLEAM J235139-894114 -0.001036 0.026022 23 51 39.45 -89 41 14.30 357.914368   0.002407 -89.687309   0.002410   0.262282 0.026270   0.248581  2.95339e-02  2.19263e+02  1.11849e+01 146.4811    5.638908  -4.158033    0.446378 -0.011716 0.015203 80 3 221.397 152.189  22.430508  0.116048 0.230260    0.531743 0.199787    0.528997 0.198755  5.18718e+02 443.560  -4.158033  0.150878  0.245333 519.623 445.444  14.269532  0.064158 0.170919    0.512256  0.152039    0.509285  0.151157  4.68570e+02 376.896  -4.158033  0.031757 0.134961 469.572 379.126   4.874937 -0.001032 0.226090    0.427771  0.209262    0.423489  0.207168  4.40963e+02 342.630  -4.158033 -0.011465 0.171556 442.027 345.088  13.513020  0.035764 0.245972    0.444216  0.222199    0.438262  0.219220  4.00925e+02 283.699  -4.158033 -0.031922  0.304453 402.096 286.674   2.319414  0.000086 0.097613    0.332228  0.087342    0.326553  0.085850  3.66049e+02 271.377  -4.158033  0.016291 0.086581 367.331 274.487   9.087766  0.007765 0.067661    0.296139  0.059777    0.291348  0.058810  3.32573e+02 256.153  -4.158033  0.034249 0.100245 333.983 259.448   9.828724  0.022250 0.069156    0.202653  0.061831    0.198947  0.060700  3.16835e+02 244.937  -4.158033  0.010231 0.068757 318.315 248.382   2.127662  0.012362 0.052444    0.298724  0.046800    0.292667  0.045851  3.02262e+02 230.963  -4.158033 -0.013592 0.076489 303.812 234.615  22.214157  0.008152 0.052375    0.341529  0.048493    0.331788  0.047110  2.67819e+02 183.996  -4.158033  0.006087 0.070793 269.568 188.566  20.953054  0.001960 0.050528    0.276829  0.045727    0.267163  0.044130  2.54018e+02 171.133  -4.158033  0.012933 0.094522 255.862 176.038  22.808126  0.024398 0.071617    0.248030  0.064988    0.239452  0.062741  2.46127e+02 170.159  -4.158033  0.029935 0.116816 248.029 175.092  12.885068  0.007915 0.056775    0.278406  0.052344    0.268323  0.050448  2.28098e+02 165.932  -4.158033 -0.003216 0.038754 230.149 170.987  19.574308  0.008273 0.050534    0.345853  0.045476    0.330670  0.043480  2.27678e+02 148.799  -4.158033 -0.000142 0.058509 229.733 154.417  21.433491  0.007071 0.072735    0.317384  0.065278    0.303373  0.062396  2.16998e+02 141.9525  -4.158033 -0.003245 0.054701 219.153 147.832  10.051599  0.001770 0.058237    0.320320  0.052301    0.304456  0.049711  2.13273e+02 138.8363  -4.158033 -0.005387 0.040263 215.466 144.842   6.488752 -0.004338 0.059530    0.179936  0.053834    0.168986  0.050558  2.07908e+02 130.4126  -4.158033 -0.015960 0.085525 210.156 136.790  12.743345  0.004307 0.070833    0.205871  0.064479    0.194392  0.060883  1.88324e+02 134.0238  -4.158033 -0.069456 0.078095 190.803 140.237  14.509152 -0.007659 0.112504    0.495948  0.101910    0.464646  0.095478  1.72234e+02 125.0827  -4.158033 -0.014896 0.124724 174.942 131.7186   7.999518 -0.009265 0.082080    0.167417  0.074100    0.156235  0.069151  1.71534e+02 127.1215  -4.158033 -0.038894 0.098901 174.253 133.6561  17.220127 -0.008993 0.063705    0.097147  0.058071    0.090779  0.054264  1.70630e+02 123.0671  -4.158033  0.010692 0.061223 173.363 129.8062  18.959024 -0.370882  0.206630   1.327554   0.271901  0.282624
""".strip()  # noqa: E501


@pytest.fixture
def phasecentre():
    phasecentre = MagicMock(name="phasecentre")
    phasecentre.ra.radian = 350 * np.pi / 180
    phasecentre.dec.radian = -85.0 * np.pi / 180
    return phasecentre


@pytest.fixture
def gleam_open():
    """Patch ``open`` to yield the given GLEAM lines."""
    with patch("builtins.open") as open_mock:
        fileobj = MagicMock(name="gleamfile object")
        fileobj.__iter__.return_value = [GLEAMFILE_LINE]

        open_mock.return_value = open_mock
        open_mock.__enter__.return_value = fileobj
        yield open_mock, fileobj


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.sky_model_reader.logger"
)
def test_read_gleamegc(logger_mock, gleam_open, phasecentre):
    """
    Test that read_gleamegc reads the catalogue file line by line,
    and returns a SkyModel, where each component satisfies
    following criterions:

    1. The integrated flux of the component in the wide range is greater
    than minimum flux density

    2. The component is within the field of view
    """
    open_mock, fileobj = gleam_open

    expected_component = SkyComponent(
        component_id="J235139-894114",
        source_id="",
        ra_deg=357.914368,
        dec_deg=-89.687309,
        i_pol_jy=0.271901,
        ref_freq_hz=200000000.0,
        epoch=0.0,
        # GLEAM axes are FWHM, halved to semi-axes
        a_arcsec=219.263 / 2,
        b_arcsec=146.4811 / 2,
        pa_deg=-4.158033,
        spec_idx=[-0.370882],
        log_spec_idx=True,
    )

    # mock deconvolve_gaussian to return the parsed shape parameters
    with patch(
        "ska_sdp_instrumental_calibration.data_managers"
        ".sky_model.sky_model_reader.deconvolve_gaussian",
        side_effect=lambda a, b, pa, *_: (a, b, pa),
    ) as deconvolve_gaussian_mock:
        sky_model = read_gleamegc(
            "gleamfile.dat", phase_centre=phasecentre, fov_deg=10
        )
        deconvolve_gaussian_mock.assert_called_once_with(
            219.263 / 2,
            146.4811 / 2,
            -4.158033,
            221.397 / 2,
            152.189 / 2,
            22.430508,
        )

    open_mock.assert_called_once_with("gleamfile.dat", "r")

    logger_mock.info.assert_called_once_with("extracted 1 GLEAM components")
    logger_mock.debug.assert_called_once_with(
        "alpha: mean = -0.37, median = -0.37, used = -0.78"
    )

    assert sky_model.components == {"J235139-894114": expected_component}


def test_read_gleamegc_sets_metadata(gleam_open):
    sky_model = read_gleamegc("gleamfile.dat")

    metadata = sky_model.metadata
    assert metadata.catalogue_name == "GLEAM-EGC"
    assert metadata.version == "1.0.0"
    assert metadata.reference == "https://doi.org/10.1093/mnras/stw2337"
    assert metadata.freq_min_hz == 72e6
    assert metadata.freq_max_hz == 231e6

    uploaded_at = datetime.fromisoformat(metadata.uploaded_at)
    assert uploaded_at.tzinfo == timezone.utc


def test_read_gleamegc_does_not_filter_by_fov_without_phase_centre(
    gleam_open,
):
    sky_model = read_gleamegc("gleamfile.dat")

    assert list(sky_model.components) == ["J235139-894114"]


def test_read_gleamegc_raises_when_fov_set_without_phase_centre():
    with pytest.raises(ValueError, match="phase_centre must be provided"):
        read_gleamegc("gleamfile.dat", fov_deg=10)


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.sky_model_reader.logger"
)
def test_should_exclude_component_when_flux_is_less_than_min_flux(
    logger_mock, gleam_open, phasecentre
):
    """
    Test that for a Component in the gleam catalogue file,
    if its integrated flux in the wide range is less than
    minimum flux density, then its excluded from the list of components.
    """
    open_mock, fileobj = gleam_open

    sky_model = read_gleamegc(
        "gleamfile.dat",
        phase_centre=phasecentre,
        fov_deg=5.0,
        flux_limit=1.0,
    )

    open_mock.assert_called_once_with("gleamfile.dat", "r")

    logger_mock.info.assert_called_once_with("extracted 0 GLEAM components")
    logger_mock.debug.assert_not_called()

    assert sky_model.components == {}


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.sky_model_reader.logger"
)
def test_should_set_flux_alpha_to_defaults_when_fitted_data_is_unspecified(
    logger_mock, gleam_open, phasecentre
):
    """
    Test that for a Component in the gleam catalogue file,
    if its integrated flux at 200 Mhz is unspecified, then:

    1. flux of the component is set to its integrated flux in the wide range

    2. spectral index (alpha) of the component is set to the nominal
    spectral index value (passed as a parameter)
    """
    _, fileobj = gleam_open
    line = GLEAMFILE_LINE[:3135] + "  ---     " + GLEAMFILE_LINE[3145:]
    fileobj.__iter__.return_value = [line]

    with patch(
        "ska_sdp_instrumental_calibration.data_managers"
        ".sky_model.sky_model_reader.deconvolve_gaussian",
        return_value=(219.263, 146.4811, -4.158033),
    ) as deconvolve_gaussian_mock:
        sky_model = read_gleamegc(
            "gleamfile.dat",
            phase_centre=phasecentre,
            fov_deg=10,
            alpha0=-0.65,
        )
        deconvolve_gaussian_mock.assert_called_once()

    logger_mock.info.assert_called_once_with("extracted 1 GLEAM components")
    logger_mock.debug.assert_not_called()

    # flux and alpha are set to Fintwide and alpha0 respectively
    component = sky_model.components["J235139-894114"]
    assert component.i_pol_jy == 0.248581
    assert component.spec_idx == [-0.65]
    assert component.log_spec_idx is True


def test_exclude_component_when_its_out_of_fov(gleam_open, phasecentre):
    """
    Test that for a Component in the gleam catalogue file,
    if its position is far away from the phase centre (outside of the
    field of view), then its excluded from the final list of components.
    """
    sky_model = read_gleamegc(
        "gleamfile.dat", phase_centre=phasecentre, fov_deg=5.0
    )

    assert sky_model.components == {}


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.sky_model_reader.LocalSkyModel"
)
def test_should_export_lsm_to_csv(local_skymodel_mock):
    local_skymodel_mock.return_value = local_skymodel_mock
    components = [
        SkyComponent(
            component_id="GLEAM J000010-000001",
            source_id="",
            epoch=0.0,
            ra_deg=357.914368,
            dec_deg=-89.687309,
            i_pol_jy=0.271901,
            ref_freq_hz=200000000.0,
            spec_idx=[-0.7, 0.01, 0.123],
            a_arcsec=219.263,
            b_arcsec=146.4811,
            pa_deg=-4.158033,
            log_spec_idx=True,
        )
    ]
    export_lsm_to_csv(components, "./test.csv")
    local_skymodel_mock.assert_called_once_with(
        column_names=[
            "component_id",
            "source_id",
            "ra_deg",
            "dec_deg",
            "i_pol_jy",
            "ref_freq_hz",
            "epoch",
            "a_arcsec",
            "b_arcsec",
            "pa_deg",
            "spec_idx",
            "log_spec_idx",
        ],
        num_rows=1,
        vector_columns=["spec_idx"],
    )
    local_skymodel_mock.set_row.assert_called_once_with(
        0,
        {
            "component_id": "GLEAM J000010-000001",
            "source_id": "",
            "ref_freq_hz": 200000000.0,
            "ra_deg": 357.914368,
            "dec_deg": -89.687309,
            "i_pol_jy": 0.271901,
            "epoch": 0.0,
            "a_arcsec": 219.263,
            "b_arcsec": 146.4811,
            "pa_deg": -4.158033,
            "spec_idx": [-0.7, 0.01, 0.123],
            "log_spec_idx": True,
        },
    )
    local_skymodel_mock.save.assert_called_once_with("./test.csv")


class TestComponentConverts:
    def test_should_create_sky_components_from_df(self):
        columns = [
            "component_id",
            "source_id",
            "epoch",
            "ra_deg",
            "dec_deg",
            "i_pol_jy",
            "a_arcsec",
            "b_arcsec",
            "pa_deg",
            "ref_freq_hz",
            "spec_idx",
            "log_spec_idx",
        ]

        # Data provided as a list of lists
        data = [
            [
                "Component 000000",
                "source 1",
                2026.2247,
                197.914612,
                -22.277973,
                45.91633,
                0.0,
                0.0,
                0.0,
                1.43e08,
                [0.0],
                True,
            ],
            [
                "Component 000001",
                "source 2",
                2026.232,
                198.034622,
                -20.447077,
                3.442318,
                0.0,
                0.0,
                0.0,
                1.43e08,
                [-0.6756219],
                True,
            ],
        ]

        # Create the DataFrame
        df = pd.DataFrame(data=data, columns=columns)
        expected = [
            SkyComponent(
                component_id="Component 000000",
                source_id="source 1",
                epoch=2026.2247,
                ra_deg=197.914612,
                dec_deg=-22.277973,
                i_pol_jy=45.91633,
                ref_freq_hz=1.43e08,
                spec_idx=[0.0],
                a_arcsec=0.0,
                b_arcsec=0.0,
                pa_deg=0.0,
                log_spec_idx=True,
            ),
            SkyComponent(
                component_id="Component 000001",
                source_id="source 2",
                epoch=2026.232,
                ra_deg=198.034622,
                dec_deg=-20.447077,
                i_pol_jy=3.442318,
                ref_freq_hz=1.43e08,
                spec_idx=[-0.6756219],
                a_arcsec=0.0,
                b_arcsec=0.0,
                pa_deg=0.0,
                log_spec_idx=True,
            ),
        ]

        assert expected == ComponentConverters.df_to_components(df)
