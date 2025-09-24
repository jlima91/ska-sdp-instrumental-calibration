import re

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from mock import MagicMock, Mock, call, patch
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

from ska_sdp_instrumental_calibration.processing_tasks.lsm import Component
from ska_sdp_instrumental_calibration.processing_tasks.predict_model.predict import (  # noqa: E501
    convert_comp_to_skycomponent,
    correct_comp_vis_ufunc,
    dft_skycomponent_ufunc,
    gaussian_tapers_ufunc,
    generate_rotation_matrices,
    predict_vis,
)


class TestPredictVis:
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.dft_skycomponent_ufunc"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.with_chunks"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.convert_comp_to_skycomponent"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.xr.apply_ufunc"
    )
    def test_predict_from_components(
        self,
        xr_apply_ufunc_mock,
        convert_comp_to_skycomponent_mock,
        with_chunks_mock,
        dft_skycomponent_ufunc_mock,
        generate_vis,
    ):
        vis, _ = generate_vis

        with_chunks_mock.return_value = vis.frequency
        xr_apply_ufunc_mock.side_effect = [vis.vis, vis.vis]

        skycomponent_mock_1 = MagicMock(name="skycomponent-1")
        skycomponent_mock_1.flux = 0.4
        skycomponent_mock_2 = MagicMock(name="skycomponent-2")
        skycomponent_mock_2.flux = 0.8

        convert_comp_to_skycomponent_mock.side_effect = [
            skycomponent_mock_1,
            skycomponent_mock_2,
        ]

        com_mock_1 = (MagicMock(name="component-1"),)
        com_mock_2 = (MagicMock(name="component-2"),)
        lsm = [com_mock_1, com_mock_2]

        expected = vis.vis + vis.vis

        uvw_sel_mock = Mock(name="uvw-sel")
        uvw_sel_mock.sel.return_value = "uvw_sel"

        uvw_mock = MagicMock(name="uvw")
        uvw_mock.__truediv__.return_value = uvw_sel_mock
        uvw_mock.__mul__.return_value = uvw_mock

        actual = predict_vis(
            vis.vis,
            uvw_mock,
            vis.datetime,
            vis.configuration,
            vis.antenna1,
            vis.antenna2,
            lsm,  # type: ignore
            vis.phasecentre,
            beam_type=None,
        )

        np.testing.assert_equal(actual.data, expected.data)

        args = convert_comp_to_skycomponent_mock.call_args_list

        np.testing.assert_equal(args[0].args[0], com_mock_1)
        np.testing.assert_equal(args[0].args[1].data, vis.frequency.data)
        np.testing.assert_equal(args[0].args[2].data, vis.polarisation.data)

        np.testing.assert_equal(args[1].args[0], com_mock_2)

        assert convert_comp_to_skycomponent_mock.call_count == 2

        xr_apply_ufunc_mock.assert_has_calls(
            [
                call(
                    dft_skycomponent_ufunc_mock,
                    "uvw_sel",
                    "uvw_sel",
                    "uvw_sel",
                    0.4,
                    input_core_dims=[
                        [
                            "baselineid",
                        ],
                        [
                            "baselineid",
                        ],
                        [
                            "baselineid",
                        ],
                        ["polarisation"],
                    ],
                    output_core_dims=[("baselineid", "polarisation")],
                    dask="parallelized",
                    output_dtypes=[vis.vis.dtype],
                    kwargs={
                        "skycomponent": skycomponent_mock_1,
                        "phase_centre": vis.phasecentre,
                    },
                ),
                call(
                    dft_skycomponent_ufunc_mock,
                    "uvw_sel",
                    "uvw_sel",
                    "uvw_sel",
                    0.8,
                    input_core_dims=[
                        [
                            "baselineid",
                        ],
                        [
                            "baselineid",
                        ],
                        [
                            "baselineid",
                        ],
                        ["polarisation"],
                    ],
                    output_core_dims=[("baselineid", "polarisation")],
                    dask="parallelized",
                    output_dtypes=[vis.vis.dtype],
                    kwargs={
                        "skycomponent": skycomponent_mock_2,
                        "phase_centre": vis.phasecentre,
                    },
                ),
            ]
        )

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.correct_comp_vis_ufunc"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.AltAz"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.create_beams"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.with_chunks"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.convert_comp_to_skycomponent"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.xr.apply_ufunc"
    )
    def test_predict_from_components_with_everybeam(
        self,
        xr_apply_ufunc_mock,
        convert_comp_to_skycomponent_mock,
        with_chunks_mock,
        create_beams_mock,
        AltAz_mock,
        process_comp_vis_and_responce_mock,
        generate_vis,
    ):
        vis, _ = generate_vis

        with_chunks_mock.return_value = vis.frequency
        xr_apply_ufunc_mock.side_effect = [vis.vis, vis.vis, vis.vis, vis.vis]

        AltAz_mock.alt.degree = 1
        AltAz_mock.return_value = AltAz_mock

        skycomponent_mock = MagicMock(name="skycomponent-1")
        skycomponent_mock.flux = 0.4
        mock_direction = MagicMock(name="direction")
        mock_direction.transform_to.return_value = AltAz_mock
        skycomponent_mock.direction = mock_direction
        convert_comp_to_skycomponent_mock.return_value = skycomponent_mock

        com_mock_1 = (MagicMock(name="component-1"),)
        com_mock_2 = (MagicMock(name="component-2"),)

        lsm = [com_mock_1, com_mock_2]

        beams_mocked = MagicMock(name="beams")
        beam_reasponse_mock = MagicMock(name="response-xda")
        beam_reasponse_mock.assign_coords.return_value = beam_reasponse_mock
        beam_reasponse_mock.pipe.return_value = beam_reasponse_mock

        beams_mocked.array_response.return_value = beam_reasponse_mock
        create_beams_mock.return_value = beams_mocked

        expected = vis.vis * 2

        actual = predict_vis(
            vis.vis,
            vis.uvw,
            vis.datetime,
            vis.configuration,
            vis.antenna1,
            vis.antenna2,
            lsm,  # type: ignore
            vis.phasecentre,
            beam_type="everybeam",
            eb_coeffs="coeffs",
            eb_ms="mspath",
        )

        np.testing.assert_equal(actual.data, expected.data)

        time = np.mean(Time(vis.datetime.data))  # type: ignore
        assert create_beams_mock.call_args.args[0] == time
        np.testing.assert_equal(
            create_beams_mock.call_args.args[1].data, vis.frequency.data
        )
        assert create_beams_mock.call_args.args[2] == vis.configuration
        assert create_beams_mock.call_args.args[3] == vis.phasecentre
        assert create_beams_mock.call_args.args[4] == "coeffs"
        assert create_beams_mock.call_args.args[5] == "mspath"

        mock_direction.transform_to.assert_has_calls(
            [call(AltAz_mock), call(AltAz_mock)]
        )

        args = beams_mocked.array_response.call_args_list

        np.testing.assert_equal(args[0].kwargs["direction"], mock_direction)
        np.testing.assert_equal(
            args[0].kwargs["frequency_xdr"].data, vis.frequency.data
        )
        np.testing.assert_equal(args[0].kwargs["time"], time)

        assert beams_mocked.array_response.call_count == 2

        np.testing.assert_equal(
            beam_reasponse_mock.assign_coords.call_args_list[0]
            .args[0]["id"]
            .data,
            vis.configuration.id.data,
        )

        beam_reasponse_mock.pipe.assert_has_calls(
            [
                call(with_chunks_mock, vis.chunksizes),
                call(with_chunks_mock, vis.chunksizes),
            ]
        )

        args = xr_apply_ufunc_mock.mock_calls[1].args
        assert args[0] == process_comp_vis_and_responce_mock
        assert args[2] == beam_reasponse_mock
        np.testing.assert_equal(args[1].data, vis.vis)
        np.testing.assert_equal(args[3].data, vis.antenna1)
        np.testing.assert_equal(args[4].data, vis.antenna2)

        args = xr_apply_ufunc_mock.mock_calls[3].args
        assert args[0] == process_comp_vis_and_responce_mock
        assert args[2] == beam_reasponse_mock
        np.testing.assert_equal(args[1].data, vis.vis)
        np.testing.assert_equal(args[3].data, vis.antenna1)
        np.testing.assert_equal(args[4].data, vis.antenna2)

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.correct_comp_vis_ufunc"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.generate_rotation_matrices"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.dft_skycomponent_ufunc"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.with_chunks"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.convert_comp_to_skycomponent"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.xr.apply_ufunc"
    )
    def test_predict_from_components_with_station_rm(
        self,
        xr_apply_ufunc_mock,
        convert_comp_to_skycomponent_mock,
        with_chunks_mock,
        dft_skycomponent_ufunc_mock,
        generate_rotation_matrices_mock,
        process_comp_vis_and_responce_mock,
        generate_vis,
    ):
        vis, _ = generate_vis

        rot_arr_mock = MagicMock(name="rot_arr_mock")
        rot_arr_mock.chunk.return_value = rot_arr_mock
        generate_rotation_matrices_mock.return_value = rot_arr_mock

        with_chunks_mock.return_value = vis.frequency
        xr_apply_ufunc_mock.side_effect = [vis.vis, vis.vis, vis.vis, vis.vis]

        skycomponent_mock_1 = MagicMock(name="skycomponent-1")
        skycomponent_mock_1.flux = 0.4
        skycomponent_mock_2 = MagicMock(name="skycomponent-2")
        skycomponent_mock_2.flux = 0.8

        convert_comp_to_skycomponent_mock.side_effect = [
            skycomponent_mock_1,
            skycomponent_mock_2,
        ]

        com_mock_1 = (MagicMock(name="component-1"),)
        com_mock_2 = (MagicMock(name="component-2"),)
        lsm = [com_mock_1, com_mock_2]

        expected = vis.vis * 2

        actual = predict_vis(
            vis.vis,
            vis.uvw,
            vis.datetime,
            vis.configuration,
            vis.antenna1,
            vis.antenna2,
            lsm,  # type: ignore
            vis.phasecentre,
            beam_type=None,
            station_rm=np.zeros(vis.configuration.id.shape),
        )

        np.testing.assert_equal(actual.data, expected.data)

        args = xr_apply_ufunc_mock.mock_calls[1].args
        assert args[0] == process_comp_vis_and_responce_mock
        assert args[2] == rot_arr_mock
        np.testing.assert_equal(args[1].data, vis.vis)
        np.testing.assert_equal(args[3].data, vis.antenna1)
        np.testing.assert_equal(args[4].data, vis.antenna2)

        args = xr_apply_ufunc_mock.mock_calls[3].args
        assert args[0] == process_comp_vis_and_responce_mock
        assert args[2] == rot_arr_mock
        np.testing.assert_equal(args[1].data, vis.vis)
        np.testing.assert_equal(args[3].data, vis.antenna1)
        np.testing.assert_equal(args[4].data, vis.antenna2)

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.AltAz"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.create_beams"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.correct_comp_vis_ufunc"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.generate_rotation_matrices"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.dft_skycomponent_ufunc"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.with_chunks"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ".predict_model.predict.convert_comp_to_skycomponent"
    )
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ""
        ".predict_model.predict.xr.apply_ufunc"
    )
    def test_predict_from_components_with_everybeam_station_rm(
        self,
        xr_apply_ufunc_mock,
        convert_comp_to_skycomponent_mock,
        with_chunks_mock,
        dft_skycomponent_ufunc_mock,
        generate_rotation_matrices_mock,
        process_comp_vis_and_responce_mock,
        create_beams_mock,
        AltAz_mock,
        generate_vis,
    ):
        vis, _ = generate_vis

        beams_mocked = MagicMock(name="beams")
        beam_reasponse_mock = MagicMock(name="response-xda")
        beam_reasponse_mock.assign_coords.return_value = beam_reasponse_mock
        beam_reasponse_mock.pipe.return_value = beam_reasponse_mock

        beams_mocked.array_response.return_value = beam_reasponse_mock
        create_beams_mock.return_value = beams_mocked

        rot_arr_mock = MagicMock(name="rot_arr_mock")
        rot_arr_mock.chunk.return_value = rot_arr_mock
        generate_rotation_matrices_mock.return_value = rot_arr_mock

        with_chunks_mock.return_value = vis.frequency
        xr_apply_ufunc_mock.side_effect = [
            vis.vis,
            vis.vis,
            vis.vis,
            vis.vis,
            vis.vis,
            vis.vis,
        ]

        AltAz_mock.alt.degree = 1
        AltAz_mock.return_value = AltAz_mock

        skycomponent_mock = MagicMock(name="skycomponent-1")
        skycomponent_mock.flux = 0.4
        mock_direction = MagicMock(name="direction")
        mock_direction.transform_to.return_value = AltAz_mock
        skycomponent_mock.direction = mock_direction
        convert_comp_to_skycomponent_mock.return_value = skycomponent_mock

        convert_comp_to_skycomponent_mock.return_value = skycomponent_mock

        com_mock_1 = (MagicMock(name="component-1"),)
        com_mock_2 = (MagicMock(name="component-2"),)
        lsm = [com_mock_1, com_mock_2]

        expected = vis.vis * 2

        actual = predict_vis(
            vis.vis,
            vis.uvw,
            vis.datetime,
            vis.configuration,
            vis.antenna1,
            vis.antenna2,
            lsm,  # type: ignore
            vis.phasecentre,
            beam_type="everybeam",
            eb_coeffs="coeffs",
            eb_ms="mspath",
            station_rm=np.zeros(vis.configuration.id.shape),
        )

        np.testing.assert_equal(actual.data, expected.data)

        # making sure apply_ufunc is called alteast three times
        # with three different functions.

        # occurence 1
        args = xr_apply_ufunc_mock.mock_calls[0].args
        assert args[0] == np.matmul
        assert args[1] == beam_reasponse_mock
        assert args[2] == rot_arr_mock

        # occurence 2
        args = xr_apply_ufunc_mock.mock_calls[1].args
        assert args[0] == dft_skycomponent_ufunc_mock

        # occurence 3
        args = xr_apply_ufunc_mock.mock_calls[2].args
        assert args[0] == process_comp_vis_and_responce_mock


class TestConvertCompToSkycomponent:
    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ""
        ".predict_model.predict.deconvolve_gaussian"
    )
    def test_convert_comp_to_skycomponents(self, deconvolve_gaussian_mock):
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
        freq = np.array([400, 800])
        frequency_xdr = xr.DataArray(freq, coords={"frequency": freq})

        pol = np.array(["XX", "XY", "YX", "YY"])
        pol_coord = xr.DataArray(pol, coords={"polarisation": pol})

        skycomp = convert_comp_to_skycomponent(
            component, frequency_xdr, pol_coord
        )

        deconvolve_gaussian_mock.assert_called_once_with(component)

        # SkyComponent does not implement @dataclass or __equal__
        actual_component = skycomp
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

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks"
        ""
        ".predict_model.predict.deconvolve_gaussian"
    )
    def test_convert_point_source_to_skycomponent(
        self, deconvolve_gaussian_mock
    ):
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

        freq = np.array([400, 800])
        frequency_xdr = xr.DataArray(freq, coords={"frequency": freq})

        pol = np.array(["XX", "XY", "YX", "YY"])
        pol_coord = xr.DataArray(pol, coords={"polarisation": pol})

        skycomp = convert_comp_to_skycomponent(
            component, frequency_xdr, pol_coord
        )

        deconvolve_gaussian_mock.assert_called_once_with(component)

        actual_component = skycomp
        assert actual_component.shape == "POINT"
        assert actual_component.params == {}

    def test_should_throw_error_for_circular_pol(self):
        component = Component(
            name="J12345",
            RAdeg=260,
            DEdeg=-85,
            flux=4.0,
            ref_freq=200,
            alpha=2.0,
        )

        freq = np.array([400, 800])
        frequency_xdr = xr.DataArray(freq, coords={"frequency": freq})

        pol = np.array(["RR", "LL"])
        pol_coord = xr.DataArray(pol, coords={"polarisation": pol})

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Only polarisation ['XX', 'XY', 'YX', 'YY'] is supported."
            ),
        ):
            convert_comp_to_skycomponent(component, frequency_xdr, pol_coord)


def test_should_generate_rotation_matrices():
    rm = da.from_array([1, 2, 3, 4])
    config_id = [5, 6, 7, 8]
    freq = np.array([400, 800])
    frequency_xdr = xr.DataArray(freq, coords={"frequency": freq})

    expected = xr.DataArray(
        np.array(
            [
                [
                    [[-0.71936493, -0.69463235], [0.69463235, -0.71936493]],
                    [[0.03497181, 0.9993883], [-0.9993883, 0.03497181]],
                    [[0.66914067, -0.74313577], [0.74313577, 0.66914067]],
                    [[-0.99755395, 0.06990083], [-0.06990083, -0.99755395]],
                ],
                [
                    [[0.82903252, -0.5592004], [0.5592004, 0.82903252]],
                    [[0.37458982, -0.92719063], [0.92719063, 0.37458982]],
                    [[-0.20790838, -0.9781483], [0.9781483, -0.20790838]],
                    [[-0.71936493, -0.69463235], [0.69463235, -0.71936493]],
                ],
            ],
            dtype=np.float64,
        ),
        coords={"frequency": freq, "id": config_id},
        dims=("frequency", "id", "x", "y"),
    )

    actual = generate_rotation_matrices(rm, frequency_xdr, config_id)

    np.testing.assert_allclose(expected.data, actual.data)


def test_gaussian_tapers(generate_vis):
    vis, _ = generate_vis
    params = {
        "bmaj": 2 / 60.0,
        "bmin": 1 / 60.0,
        "bpa": 34,
    }

    scaled_u = np.arange(6).reshape((1, 2, 3))
    scaled_v = np.arange(6).reshape((1, 2, 3))

    expected = np.array(
        [[[1.0, 0.99999766, 0.99999062], [0.99997891, 0.9999625, 0.99994141]]]
    )
    actual = gaussian_tapers_ufunc(scaled_u, scaled_v, params)
    np.testing.assert_allclose(actual, expected)


def test_dft_point_source_skycomponent(generate_vis):
    """Test point-source component DFT."""
    vis, _ = generate_vis
    skycomp = SkyComponent(
        direction=SkyCoord(ra=-0.1, dec=-26.0, unit="deg"),
        frequency=vis.frequency.data,
        name="test-comp",
        flux=xr.DataArray(
            np.arange(16).reshape((vis.frequency.size, 4)),
            dims=("frequency", "polarisation"),
            coords={
                "polarisation": ["XX", "XY", "YY", "YX"],
                "frequency": vis.frequency,
            },
        ),
        polarisation_frame=PolarisationFrame("linear"),
        shape="POINT",
        params={},
    )
    scaled_uvw = (
        vis.uvw * vis.frequency / const.c.value  # pylint: disable=no-member
    ).transpose("time", "frequency", "baselines", "spatial")
    scaled_u = scaled_uvw.sel(spatial="u")
    scaled_v = scaled_uvw.sel(spatial="v")
    scaled_w = scaled_uvw.sel(spatial="w")

    expected_vis = vis.assign({"vis": xr.zeros_like(vis.vis)})
    dft_skycomponent_visibility(expected_vis, skycomp)

    actual_vis = dft_skycomponent_ufunc(
        scaled_u.data,
        scaled_v.data,
        scaled_w.data,
        skycomp.flux.data,
        skycomp,
        vis.phasecentre,
    )

    actual_vis = np.transpose(actual_vis, (0, 2, 1, 3))

    np.testing.assert_allclose(actual_vis, expected_vis.vis)


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks"
    ""
    ".predict_model.predict.gaussian_tapers_ufunc"
)
def test_dft_guassian_skycomponent(gaussian_tapers_ufunc_mock, generate_vis):
    """Test guassian component DFT."""
    gaussian_tapers_ufunc_mock.return_value = 1

    vis, _ = generate_vis
    skycomp = SkyComponent(
        direction=SkyCoord(ra=-0.1, dec=-26.0, unit="deg"),
        frequency=vis.frequency.data,
        name="test-comp",
        flux=xr.DataArray(
            np.arange(16).reshape((vis.frequency.size, 4)),
            dims=("frequency", "polarisation"),
            coords={
                "polarisation": ["XX", "XY", "YY", "YX"],
                "frequency": vis.frequency,
            },
        ),
        polarisation_frame=PolarisationFrame("linear"),
        shape="GAUSSIAN",
        params={},
    )
    scaled_uvw = (
        vis.uvw * vis.frequency / const.c.value  # pylint: disable=no-member
    ).transpose("time", "frequency", "baselines", "spatial")
    scaled_u = scaled_uvw.sel(spatial="u")
    scaled_v = scaled_uvw.sel(spatial="v")
    scaled_w = scaled_uvw.sel(spatial="w")

    expected_vis = vis.assign({"vis": xr.zeros_like(vis.vis)})
    dft_skycomponent_visibility(expected_vis, skycomp)

    actual_vis = dft_skycomponent_ufunc(
        scaled_u.data,
        scaled_v.data,
        scaled_w.data,
        skycomp.flux.data,
        skycomp,
        vis.phasecentre,
    )

    actual_vis = np.transpose(actual_vis, (0, 2, 1, 3))

    np.testing.assert_allclose(actual_vis, expected_vis.vis)

    call_args = gaussian_tapers_ufunc_mock.call_args_list

    np.testing.assert_allclose(call_args[0].args[0], scaled_u.data)
    np.testing.assert_allclose(call_args[0].args[1], scaled_v.data)

    assert call_args[0].args[2] == skycomp.params


def test_correct_comp_vis_ufunc():
    comp_vis = np.array([[[[0.0, 1.0, 1.0, 0.0]]]])
    correction = np.array([[[[1.0, 2.0], [1.0, 2.0]]]])
    antenna1 = np.array([0])
    antenna2 = np.array([0])

    expected = np.array([[[[4.0, 4.0, 4.0, 4.0]]]])
    actual = correct_comp_vis_ufunc(comp_vis, correction, antenna1, antenna2)

    np.testing.assert_equal(actual, expected)
