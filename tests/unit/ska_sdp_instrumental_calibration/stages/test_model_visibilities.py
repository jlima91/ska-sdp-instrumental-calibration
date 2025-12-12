from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import predict_vis_stage


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "predict_vis": {
            "beam_type": "everybeam",
            "normalise_at_beam_centre": True,
            "eb_ms": None,
            "element_response_model": "oskar_dipole_cos",
            "eb_coeffs": None,
            "gleamfile": None,
            "lsm_csv_path": None,
            "fov": 5.0,
            "flux_limit": 1.0,
            "alpha0": -0.78,
        }
    }

    assert predict_vis_stage.config == expected_config


@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".BeamsFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities.predict_vis"
)
def test_should_predict_visibilities(
    predict_vis_mock,
    global_sky_model_mock,
    beams_factory_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    cli_args = {"input": "path/to/input/ms"}
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": False,
        "eb_ms": None,
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    global_sky_model_mock.assert_called_once_with(
        upstream_output.vis.phasecentre,
        10.0,
        1.0,
        -0.78,
        "/path/to/gleam.dat",
        None,
    )
    predict_vis_mock.assert_called_once_with(
        upstream_output.vis,
        global_sky_model_mock.return_value,
        upstream_output.gaintable.time.data,
        upstream_output.gaintable.soln_interval_slices,
        beams_factory_mock.return_value,
    )

    assert result.modelvis == [1, 2, 3]


@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".BeamsFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities.predict_vis"
)
def test_should_update_call_count(
    predict_vis_mock,
    global_sky_model_mock,
    beams_factory_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    cli_args = {"input": "path/to/input/ms"}
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": False,
        "eb_ms": "test.ms",
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": None,
        "lsm_csv_path": "/path/to/lsm.csv",
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
    }

    upstream_output = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    upstream_output = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    assert upstream_output.get_call_count("predict_vis") == 2


@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".BeamsFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities.predict_vis"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".prediction_central_beams"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".apply_gaintable_to_dataset"
)
def test_should_normalise_at_beam_centre(
    apply_gaintable_mock,
    prediction_beams_mock,
    predict_vis_mock,
    global_sky_model_mock,
    beams_factory_mock,
):
    vis = Mock(name="Visibilities")
    upstream_output = UpstreamOutput()

    upstream_output["vis"] = vis
    cli_args = {"input": "path/to/input/ms"}

    model_vis = Mock(name="Model Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    predict_vis_mock.return_value = model_vis
    mock_beams = Mock(name="Beams")
    mock_beams.persist.return_value = mock_beams
    prediction_beams_mock.return_value = mock_beams

    normalised_vis = Mock(name="Normalised Visibilities")
    normalised_modelvis = Mock(name="Normalised Model Visibilities")
    apply_gaintable_mock.side_effect = [normalised_vis, normalised_modelvis]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": True,
        "eb_ms": None,
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    prediction_beams_mock.assert_called_once_with(
        upstream_output.gaintable, beams_factory_mock.return_value
    )

    apply_gaintable_mock.assert_has_calls(
        [
            call(vis, mock_beams, inverse=True),
            call(model_vis, mock_beams, inverse=True),
        ]
    )

    assert result.central_beams == mock_beams
    assert result.beams_factory == beams_factory_mock.return_value
    assert result.vis == normalised_vis
    assert result.modelvis == normalised_modelvis


@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.model_visibilities.predict_vis"
)
def test_should_perform_only_model_prediction_when_beam_type_is_not_everybeam(
    predict_vis_mock,
    global_sky_model_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    cli_args = {"input": "path/to/input/ms"}
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": None,
        "normalise_at_beam_centre": False,
        "eb_ms": None,
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    global_sky_model_mock.assert_called_once_with(
        upstream_output.vis.phasecentre,
        10.0,
        1.0,
        -0.78,
        "/path/to/gleam.dat",
        None,
    )
    predict_vis_mock.assert_called_once_with(
        upstream_output.vis,
        global_sky_model_mock.return_value,
        upstream_output.gaintable.time.data,
        upstream_output.gaintable.soln_interval_slices,
        None,
    )

    assert result.modelvis == [1, 2, 3]
