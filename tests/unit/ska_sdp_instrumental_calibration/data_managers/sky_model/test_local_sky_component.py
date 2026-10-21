import numpy as np
from astropy.coordinates import SkyCoord
from mock import ANY, Mock, patch
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    local_sky_component,
)

LocalSkyComponent = local_sky_component.LocalSkyComponent


class TestLocalSkyComponent:
    def setup_method(self):
        self.skycomponent = SkyComponent(
            component_id="J12345",
            source_id="",
            ra_deg=260,
            dec_deg=-85,
            i_pol_jy=4.0,
            ref_freq_hz=200,
            epoch=0.0,
        )

    def test_create_from_component(self):
        actual_component = LocalSkyComponent.create_from_component(
            self.skycomponent, [400, 800], "polarisation"
        )

        assert actual_component.skycomponent is self.skycomponent
        assert actual_component.polarisation_frame == PolarisationFrame(
            "linear"
        )
        np.testing.assert_allclose(
            actual_component.frequency, np.array([400, 800])
        )

    def test_direction(self):
        actual_component = LocalSkyComponent.create_from_component(
            self.skycomponent, [400, 800]
        )

        assert actual_component.direction == SkyCoord(
            ra=260, dec=-85, unit="deg"
        )

    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.dft_skycomponent"
    )
    def test_should_create_vis(self, dft_skycomponent_mock):
        lsc = LocalSkyComponent.create_from_component(
            self.skycomponent, [400, 800]
        )

        vis = lsc.create_vis("uvw", "phasecentre", "antenna1", "antenna2")

        dft_skycomponent_mock.assert_called_once_with(
            uvw="uvw",
            skycomponent=self.skycomponent,
            phase_centre="phasecentre",
            frequency=lsc.frequency,
            polarisation_frame=PolarisationFrame("linear"),
        )
        assert vis == dft_skycomponent_mock.return_value

    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.apply_antenna_gains_to_visibility"
    )
    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.dft_skycomponent"
    )
    def test_should_create_vis_with_beam_and_no_faraday(
        self, dft_skycomponent_mock, apply_antenna_mock
    ):
        lsc = LocalSkyComponent.create_from_component(
            self.skycomponent, [400, 800]
        )
        beam = Mock(name="beam")
        beam.array_response.return_value = np.array([1, 2])

        vis = lsc.create_vis(
            "uvw", "phasecentre", "antenna1", "antenna2", beam, None
        )

        beam.array_response.assert_called_once_with(direction=lsc.direction)
        apply_antenna_mock.assert_called_once_with(
            dft_skycomponent_mock.return_value, ANY, "antenna1", "antenna2"
        )
        np.testing.assert_array_equal(
            apply_antenna_mock.call_args.args[1], np.array([[1, 2]])
        )
        assert vis == apply_antenna_mock.return_value

    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.apply_antenna_gains_to_visibility"
    )
    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.dft_skycomponent"
    )
    def test_should_create_vis_with_faraday_and_no_beam(
        self, dft_skycomponent_mock, apply_antenna_mock
    ):
        lsc = LocalSkyComponent.create_from_component(
            self.skycomponent, [400, 800]
        )
        faraday_rot_matrix = np.array([[1], [2]])

        vis = lsc.create_vis(
            "uvw",
            "phasecentre",
            "antenna1",
            "antenna2",
            None,
            faraday_rot_matrix,
        )

        apply_antenna_mock.assert_called_once_with(
            dft_skycomponent_mock.return_value,
            faraday_rot_matrix,
            "antenna1",
            "antenna2",
        )
        assert vis == apply_antenna_mock.return_value

    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.apply_antenna_gains_to_visibility"
    )
    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.dft_skycomponent"
    )
    def test_should_create_vis_with_beam_and_faraday(
        self, dft_skycomponent_mock, apply_antenna_mock
    ):
        lsc = LocalSkyComponent.create_from_component(
            self.skycomponent, [400, 800]
        )
        beam = Mock(name="beam")
        beam.array_response.return_value = np.eye(2)[np.newaxis, ...]
        faraday_rot_matrix = 2 * np.eye(2)[np.newaxis, np.newaxis, ...]

        vis = lsc.create_vis(
            "uvw",
            "phasecentre",
            "antenna1",
            "antenna2",
            beam,
            faraday_rot_matrix,
        )

        beam.array_response.assert_called_once_with(direction=lsc.direction)
        np.testing.assert_array_equal(
            apply_antenna_mock.call_args.args[1], faraday_rot_matrix
        )
        assert vis == apply_antenna_mock.return_value
