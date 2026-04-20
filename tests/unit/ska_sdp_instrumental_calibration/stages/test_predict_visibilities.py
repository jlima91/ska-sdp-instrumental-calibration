from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import predict_vis_stage


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "predict_vis": {
            "use_everybeam": True,
            "normalise_at_beam_centre": True,
            "eb_ms": None,
            "element_response_model": "oskar_dipole_cos",
            "gleamfile": None,
            "lsm_csv_path": None,
            "fov": 5.0,
            "flux_limit": 1.0,
            "alpha0": -0.78,
            "export_sky_model": False,
        }
    }

    assert predict_vis_stage.__stage__.config == expected_config


@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".BeamsFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities.predict_vis"
)
def test_should_predict_visibilities(
    predict_vis_mock,
    global_sky_model_mock,
    beams_factory_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    input = ["path/to/input/ms"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "use_everybeam": True,
        "normalise_at_beam_centre": False,
        "eb_ms": None,
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
        "export_sky_model": False,
    }

    result = predict_vis_stage(
        upstream_output, input_ms=input, _output_dir_="./output_dir", **params
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
    beams_factory_mock.assert_called_once_with(
        nstations=upstream_output.vis.configuration.id.size,
        array_location=upstream_output.vis.configuration.location,
        direction=upstream_output.vis.phasecentre,
        ms_path=input[0],
        element_response_model="dipole_model",
    )

    assert result.modelvis == [1, 2, 3]


@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".BeamsFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities.predict_vis"
)
def test_should_update_call_count(
    predict_vis_mock,
    global_sky_model_mock,
    beams_factory_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    input = "path/to/input/ms"
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "use_everybeam": True,
        "normalise_at_beam_centre": False,
        "eb_ms": "test.ms",
        "gleamfile": None,
        "lsm_csv_path": "/path/to/lsm.csv",
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
        "export_sky_model": False,
        "_output_dir_": "./output_dir",
    }

    upstream_output = predict_vis_stage(
        upstream_output, input_ms=input, **params
    )

    upstream_output = predict_vis_stage(
        upstream_output, input_ms=input, **params
    )

    assert upstream_output.get_call_count("predict_vis") == 2


@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".BeamsFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities.predict_vis"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".prediction_central_beams"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
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
    upstream_output["ms_prefix"] = "ms_prefix"

    upstream_output["vis"] = vis
    input = "path/to/input/ms"

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
        "use_everybeam": True,
        "normalise_at_beam_centre": True,
        "eb_ms": None,
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
        "export_sky_model": False,
        "_output_dir_": "./output_dir",
    }

    result = predict_vis_stage(
        upstream_output,
        input_ms=input,
        **params,
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
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities.predict_vis"
)
def test_should_perform_only_model_prediction_when_use_everybeam_is_false(
    predict_vis_mock,
    global_sky_model_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    input = "path/to/input/ms"
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "use_everybeam": False,
        "normalise_at_beam_centre": False,
        "eb_ms": None,
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
        "export_sky_model": False,
        "_output_dir_": "./output_dir",
    }

    result = predict_vis_stage(upstream_output, input_ms=input, **params)
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


@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities"
    ".GlobalSkyModel"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.predict_visibilities.predict_vis"
)
def test_should_export_sky_model_used_for_prediction_to_csv_file(
    predict_vis_mock,
    global_sky_model_mock,
):

    global_sky_model_mock.return_value = global_sky_model_mock

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="Visibilities")
    upstream_output["gaintable"] = Mock(name="Gaintable")
    input = "path/to/input/ms"
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "use_everybeam": False,
        "normalise_at_beam_centre": False,
        "eb_ms": None,
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "element_response_model": "dipole_model",
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
        "export_sky_model": True,
    }

    predict_vis_stage(
        upstream_output, **params, _output_dir_="./output_dir", input_ms=input
    )

    global_sky_model_mock.export_sky_model_csv.assert_called_once_with(
        "./output_dir/sky_model.csv"
    )

    global_sky_model_mock.reset_mock()
    upstream_output["ms_prefix"] = "ms_prefix"
    predict_vis_stage(
        upstream_output, **params, _output_dir_="./output_dir", input_ms=input
    )

    global_sky_model_mock.export_sky_model_csv.assert_called_once_with(
        "./output_dir/ms_prefix_sky_model.csv"
    )
