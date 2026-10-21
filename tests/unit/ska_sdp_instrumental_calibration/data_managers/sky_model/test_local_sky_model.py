import numpy as np
from dask.base import tokenize
from mock import ANY, MagicMock, call, patch
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    local_sky_model,
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
    ".sky_model.local_sky_model.generate_lsm_from_csv"
)
def test_should_get_local_sky_model_with_components_above_horizon(
    generate_lsm_from_csv_mock, convert_time_mock, is_above_horizon_mock
):
    components = [
        make_sky_component("A", 1.0, 2.0, 3.0),
        make_sky_component("B", 4.0, 5.0, 6.0),
    ]
    generate_lsm_from_csv_mock.return_value = components
    is_above_horizon_mock.side_effect = [False, True]
    array_location = MagicMock(name="array_location")

    gsm = GlobalSkyModel("phasecentre", lsm_csv_path="sky_model.csv")
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
