import numpy as np
from astropy.coordinates import SkyCoord
from mock import ANY, Mock, call, patch
from ska_sdp_datamodels.science_data_model import PolarisationFrame

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    Component,
    local_sky_model,
    sky_model_reader,
)

GlobalSkyModel = local_sky_model.GlobalSkyModel
LocalSkyComponent = local_sky_model.LocalSkyComponent


SKY_MODEL_CSV_HEADER = sky_model_reader.SKY_MODEL_CSV_HEADER


class TestLocalSkyComponent:
    def test_create_skycomponent_from_component(self):
        component = Component(
            component_id="J12345",
            ra=260,
            dec=-85,
            i_pol=4.0,
            ref_freq=200,
            spec_idx=[2.0],
        )

        component.deconvolve_gaussian = Mock(
            name="deconvolve_gaussian", return_value=(7200, 9000, 5.0)
        )

        actual_component = LocalSkyComponent.create_from_component(
            component, [400, 800]
        )

        component.deconvolve_gaussian.assert_called_once()

        assert actual_component.direction == SkyCoord(
            ra=260, dec=-85, unit="deg"
        )
        assert actual_component.name == "J12345"
        assert actual_component.polarisation_frame == PolarisationFrame(
            "linear"
        )
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

    def test_create_skycomponent_from_point_source(self):
        component = Component(
            component_id="J12345",
            ra=260,
            dec=-85,
            i_pol=4.0,
            ref_freq=200,
            spec_idx=[2.0],
        )

        component.deconvolve_gaussian = Mock(
            name="deconvolve_gaussian", return_value=(0, 0, 0)
        )

        actual_component = LocalSkyComponent.create_from_component(
            component, [400, 800]
        )

        component.deconvolve_gaussian.assert_called_once()

        assert actual_component.shape == "POINT"
        assert actual_component.params == {}

    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_component.dft_skycomponent"
    )
    def test_should_create_vis(self, dft_skycomponent_mock):
        lsm = Mock(name="lsm")
        comp = LocalSkyComponent.create_vis(
            lsm, "uvw", "phasecentre", "antenna1", "antenna2"
        )

        dft_skycomponent_mock.assert_called_once_with(
            uvw="uvw", skycomponent=lsm, phase_centre="phasecentre"
        )

        assert comp == dft_skycomponent_mock.return_value

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
        lsm = Mock(name="lsm")
        beam = Mock(name="beam")
        beam.array_response.return_value = np.array([1, 2])

        comp = LocalSkyComponent.create_vis(
            lsm, "uvw", "phasecentre", "antenna1", "antenna2", beam, None
        )

        beam.array_response.assert_called_once_with(direction=lsm.direction)

        apply_antenna_mock.assert_called_once_with(
            dft_skycomponent_mock.return_value, ANY, "antenna1", "antenna2"
        )

        assert comp == apply_antenna_mock.return_value

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
        lsm = Mock(name="lsm")
        beam = Mock(name="beam")
        beam.array_response.return_value = np.array([1, 2])
        faraday_rot_matrix = np.array([[1], [2]])
        comp = LocalSkyComponent.create_vis(
            lsm,
            "uvw",
            "phasecentre",
            "antenna1",
            "antenna2",
            beam,
            faraday_rot_matrix,
        )

        beam.array_response.assert_called_once_with(direction=lsm.direction)

        apply_antenna_mock.assert_called_once_with(
            dft_skycomponent_mock.return_value, ANY, "antenna1", "antenna2"
        )

        assert comp == apply_antenna_mock.return_value


class TestGlobalSkyModel:
    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_model.write_csv"
    )
    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_model.ComponentConverters.to_csv_row"
    )
    @patch(
        "ska_sdp_instrumental_calibration.data_managers.sky_model"
        ".local_sky_model.generate_lsm_from_csv"
    )
    def test_should_export_sky_model_components_to_csv(
        self, mock_generate_lsm, mock_to_csv_row, write_csv_mock
    ):
        mock_generate_lsm.return_value = [
            "component1",
            "component2",
        ]

        rows = [
            ["row1_col1", "row1_col2"],
            ["row2_col1", "row2_col2"],
        ]

        mock_to_csv_row.side_effect = rows

        gsm = GlobalSkyModel(
            phasecentre=SkyCoord(ra=0, dec=-30, unit="deg"),
            lsm_csv_path="lsm.csv",
        )

        gsm.export_sky_model_csv("output.csv")
        mock_to_csv_row.assert_has_calls(
            [call("component1"), call("component2")]
        )

        write_csv_mock.assert_called_once_with(
            "output.csv", [SKY_MODEL_CSV_HEADER, *rows]
        )
