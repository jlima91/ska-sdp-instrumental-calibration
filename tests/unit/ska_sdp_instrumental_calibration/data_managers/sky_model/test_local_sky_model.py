from dataclasses import asdict

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from dask.base import tokenize
from mock import ANY, MagicMock, call, patch
from ska_sdp_datamodels.sky_model import (
    SkyComponent,
    SkyModel,
    SkyModelMetadata,
)

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    local_sky_model,
)
from ska_sdp_instrumental_calibration.exceptions import (
    RequiredArgumentMissingException,
)

LocalSkyModel = local_sky_model.LocalSkyModel
GlobalSkyModel = local_sky_model.GlobalSkyModel


def make_sky_component(component_id, ra_deg, dec_deg, i_pol_jy, **kwargs):
    return SkyComponent(
        component_id=component_id,
        source_id="",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        i_pol_jy=i_pol_jy,
        ref_freq_hz=200e6,
        epoch=0.0,
        log_spec_idx=True,
        **kwargs,
    )


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.LocalSkyComponent.create_from_component"
)
def test_should_create_local_sky_model(create_from_component_mock):
    components = [MagicMock(name=name) for name in ["A", "B", "C"]]
    skycomponent = MagicMock(name="skycomponent")
    skycomponent.create_vis.return_value = np.ones((1, 1, 1, 1))
    create_from_component_mock.return_value = skycomponent
    lsm = LocalSkyModel(components, 0.0)
    uvw = np.zeros((1, 1, 1))
    frequency = MagicMock(name="freq")
    frequency.size = 1

    polarisation = MagicMock(name="pol")
    polarisation.size = 1
    vis = lsm.create_vis(
        uvw, frequency, polarisation, "phasecentre", "antenna1", "antenna2"
    )

    create_from_component_mock.assert_has_calls(
        [call(comp, frequency) for comp in components]
    )

    skycomponent.create_vis.assert_has_calls(
        [call(uvw, "phasecentre", "antenna1", "antenna2", None, None)] * 3
    )

    np.testing.assert_allclose(vis, np.ones((1, 1, 1, 1)) * 3)


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.generate_rotation_matrices",
    return_value=np.ones((1,)),
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.LocalSkyComponent.create_from_component"
)
def test_should_create_local_sky_model_with_beam_and_rotation(
    create_from_component_mock, gen_rot_mat_mock
):
    beams_factory = MagicMock(name="beams_factory")
    components = [MagicMock(name=name) for name in ["A", "B", "C"]]
    skycomponent = MagicMock(name="skycomponent")
    skycomponent.create_vis.return_value = np.ones((1, 1, 1, 1))
    create_from_component_mock.return_value = skycomponent
    lsm = LocalSkyModel(components, 0.0)
    uvw = np.zeros((1, 1, 1))
    frequency = MagicMock(name="freq")
    frequency.size = 1

    polarisation = MagicMock(name="pol")
    polarisation.size = 1
    vis = lsm.create_vis(
        uvw,
        frequency,
        polarisation,
        "phasecentre",
        "antenna1",
        "antenna2",
        beams_factory,
        "station_rm",
    )

    create_from_component_mock.assert_has_calls(
        [call(comp, frequency) for comp in components]
    )

    gen_rot_mat_mock.assert_called_once_with(
        "station_rm", frequency, np.complex64
    )
    beams_factory.get_beams_low.assert_called_once_with(frequency, 0)

    skycomponent.create_vis.assert_has_calls(
        [
            call(
                uvw,
                "phasecentre",
                "antenna1",
                "antenna2",
                beams_factory.get_beams_low.return_value,
                ANY,
            )
        ]
        * 3
    )

    np.testing.assert_allclose(vis, np.ones((1, 1, 1, 1)) * 3)


def test_should_tokenize_local_sky_model_deterministically():
    components = [
        make_sky_component("A", 1.0, 2.0, 3.0, spec_idx=[-0.7]),
        make_sky_component("B", 4.0, 5.0, 6.0, spec_idx=np.array([-0.7])),
    ]
    same_components = [
        make_sky_component("A", 1.0, 2.0, 3.0, spec_idx=[-0.7]),
        make_sky_component("B", 4.0, 5.0, 6.0, spec_idx=np.array([-0.7])),
    ]
    changed_components = [
        make_sky_component("A", 1.0, 2.0, 3.0, spec_idx=[-0.7]),
        make_sky_component("B", 4.0, 5.0, 6.0, spec_idx=np.array([-0.8])),
    ]

    token = tokenize(LocalSkyModel(components, 10.0))

    assert token == tokenize(LocalSkyModel(components, 10.0))
    assert token == tokenize(LocalSkyModel(same_components, 10.0))
    assert token != tokenize(LocalSkyModel(changed_components, 10.0))
    assert token != tokenize(LocalSkyModel(components, 20.0))
    assert token != tokenize(LocalSkyModel(components[:1], 10.0))


def test_should_not_pickle_local_sky_model_while_tokenizing():
    lsm = LocalSkyModel([], 10.0)

    with patch.object(
        LocalSkyModel, "__reduce_ex__", side_effect=AssertionError
    ):
        tokenize(lsm)


@pytest.fixture
def gleamfile(tmp_path):
    path = tmp_path / "gleamfile.dat"
    path.touch()
    return str(path)


def make_sky_model(components):
    return SkyModel(
        metadata=SkyModelMetadata(
            version="1.0.0",
            catalogue_name="test",
            uploaded_at="2026-01-01T00:00:00+00:00",
        ),
        components={comp.component_id: comp for comp in components},
    )


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.read_gleamegc"
)
def test_should_read_global_sky_model_from_gleamfile(
    read_gleamegc_mock, gleamfile
):
    gsm = GlobalSkyModel(
        "phasecentre",
        fov=5.0,
        flux_limit=0.5,
        alpha0=-0.65,
        gleamfile=gleamfile,
        lsm_csv_path="sky_model.csv",
    )

    read_gleamegc_mock.assert_called_once_with(
        gleamfile,
        phase_centre="phasecentre",
        fov_deg=5.0,
        flux_limit=0.5,
        alpha0=-0.65,
    )
    assert gsm.sky_model == read_gleamegc_mock.return_value


def test_should_raise_when_gleamfile_not_found(tmp_path):
    phasecentre = SkyCoord(ra=10.0, dec=-30.0, unit="deg")

    with pytest.raises(
        FileNotFoundError, match="Cannot open gleam catalogue file"
    ):
        GlobalSkyModel(phasecentre, gleamfile=str(tmp_path / "gleamfile.dat"))


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.read_skymodel_from_lsm_csv"
)
def test_should_read_global_sky_model_from_csv(read_csv_mock, tmp_path):
    csv_path = tmp_path / "sky_model.csv"
    csv_path.touch()

    gsm = GlobalSkyModel(
        "phasecentre", fov=5.0, flux_limit=0.5, lsm_csv_path=str(csv_path)
    )

    read_csv_mock.assert_called_once_with(
        str(csv_path),
        phase_centre="phasecentre",
        fov_deg=5.0,
        min_flux_jy=0.5,
    )
    assert gsm.sky_model == read_csv_mock.return_value


def test_should_raise_when_csv_file_not_found(tmp_path):
    csv_path = tmp_path / "sky_model.csv"

    with pytest.raises(FileNotFoundError, match="Cannot open csv file"):
        GlobalSkyModel("phasecentre", lsm_csv_path=str(csv_path))


def test_should_raise_when_no_catalogue_provided():
    with pytest.raises(RequiredArgumentMissingException):
        GlobalSkyModel("phasecentre")


LSM_CSV_CONTENT = """\
# (component_id,source_id,ra_deg,dec_deg,i_pol_jy,ref_freq_hz,epoch,a_arcsec,b_arcsec,pa_deg,spec_idx,log_spec_idx) = format
# NUMBER_OF_COMPONENTS=3
# CATALOGUE_METADATA_VERSION=1.0.0
# CATALOGUE_METADATA_CATALOGUE_NAME=test
# CATALOGUE_METADATA_UPLOADED_AT=2026-01-01T00:00:00+00:00
inside,,10.0,-30.0,2.0,200000000,0,1.0,0.5,10.0,"[-0.7,0.01]",true
faint,,10.0,-30.0,0.5,200000000,0,0,0,0,"[-0.7]",true
outside,,40.0,-30.0,2.0,200000000,0,0,0,0,"[-0.7]",true
"""  # noqa: E501


def test_should_filter_csv_global_sky_model_by_fov_and_flux(tmp_path):
    csv_path = tmp_path / "sky_model.csv"
    csv_path.write_text(LSM_CSV_CONTENT)
    phasecentre = SkyCoord(ra=10.0, dec=-30.0, unit="deg")

    gsm = GlobalSkyModel(
        phasecentre, fov=10.0, flux_limit=1.0, lsm_csv_path=str(csv_path)
    )

    assert gsm.sky_model.metadata.catalogue_name == "test"
    assert list(gsm.sky_model.components) == ["inside"]
    component = gsm.sky_model.components["inside"]
    assert component.i_pol_jy == 2.0
    assert component.a_arcsec == 1.0
    assert component.b_arcsec == 0.5
    assert component.pa_deg == 10.0
    np.testing.assert_array_equal(component.spec_idx[:2], [-0.7, 0.01])
    assert component.log_spec_idx


def test_should_export_global_sky_model_csv_and_read_it_back(tmp_path):
    csv_path = tmp_path / "sky_model.csv"
    csv_path.write_text(LSM_CSV_CONTENT)
    phasecentre = SkyCoord(ra=10.0, dec=-30.0, unit="deg")
    gsm = GlobalSkyModel(
        phasecentre, fov=100.0, flux_limit=0.0, lsm_csv_path=str(csv_path)
    )
    output_path = tmp_path / "output.csv"

    gsm.export_sky_model_csv(str(output_path))
    exported = GlobalSkyModel(
        phasecentre, fov=100.0, flux_limit=0.0, lsm_csv_path=str(output_path)
    )

    assert exported.sky_model.metadata == gsm.sky_model.metadata
    assert list(exported.sky_model.components) == [
        "inside",
        "faint",
        "outside",
    ]
    for comp_id, component in gsm.sky_model.components.items():
        np.testing.assert_equal(
            asdict(exported.sky_model.components[comp_id]), asdict(component)
        )


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.is_above_horizon"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.convert_time_to_solution_time"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.read_gleamegc"
)
def test_should_get_local_sky_model_with_components_above_horizon(
    read_gleamegc_mock, convert_time_mock, is_above_horizon_mock, gleamfile
):
    components = [
        make_sky_component("A", 1.0, 2.0, 3.0),
        make_sky_component("B", 4.0, 5.0, 6.0),
    ]
    read_gleamegc_mock.return_value = make_sky_model(components)
    is_above_horizon_mock.side_effect = [False, True]
    array_location = MagicMock(name="array_location")

    gsm = GlobalSkyModel("phasecentre", gleamfile=gleamfile)
    lsm = gsm.get_local_sky_model(10.0, array_location)

    convert_time_mock.assert_called_once_with(10.0)
    is_above_horizon_mock.assert_has_calls(
        [
            call(comp, convert_time_mock.return_value, array_location)
            for comp in components
        ]
    )
    assert lsm.components == [components[1]]
    assert lsm.soln_time == 10.0
