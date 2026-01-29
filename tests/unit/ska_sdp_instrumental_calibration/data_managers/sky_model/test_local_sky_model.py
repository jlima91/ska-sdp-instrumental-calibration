import numpy as np
from mock import ANY, MagicMock, call, patch

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    local_sky_model,
)

LocalSkyModel = local_sky_model.LocalSkyModel


@patch(
    "ska_sdp_instrumental_calibration.data_managers"
    ".sky_model.local_sky_model.LocalSkyComponent.create_from_component"
)
def test_should_create_local_sky_model(create_from_component_mock):
    components = ["A", "B", "C"]
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
        [
            call("A", frequency, polarisation),
            call("B", frequency, polarisation),
            call("C", frequency, polarisation),
        ]
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
    components = ["A", "B", "C"]
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
        [
            call("A", frequency, polarisation),
            call("B", frequency, polarisation),
            call("C", frequency, polarisation),
        ]
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
