import pytest
from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.exceptions import (
    RequiredArgumentMissingException,
)
from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import predict_vis_stage


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".get_phasecentre"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".generate_lsm_from_gleamegc"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".predict_vis"
)
def test_should_predict_visibilities(
    predict_vis_mock, generate_lsm_mock, get_phasecentre_mock
):

    upstream_output = UpstreamOutput()
    vis = Mock(name="Visibilities")
    vis.assign.return_value = [1, 2, 3]
    upstream_output["vis"] = vis
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": False,
        "eb_ms": None,
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "fov": 10.0,
        "flux_limit": 1.0,
        "export_model_vis": False,
        "alpha0": -0.78,
        "reset_vis": False,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    get_phasecentre_mock.assert_called_once_with("path/to/input/ms")
    generate_lsm_mock.assert_called_once_with(
        gleamfile="/path/to/gleam.dat",
        phasecentre=(0.0, 0.0),
        fov=10.0,
        flux_limit=1.0,
        alpha0=-0.78,
    )
    predict_vis_mock.assert_called_once_with(
        upstream_output.vis.vis,
        upstream_output.vis.uvw,
        upstream_output.vis.datetime,
        upstream_output.vis.configuration,
        upstream_output.vis.antenna1,
        upstream_output.vis.antenna2,
        ["source1", "source2"],
        upstream_output.vis.phasecentre,
        beam_type="everybeam",
        eb_ms="path/to/input/ms",
        eb_coeffs="/path/to/coeffs",
    )

    assert result.modelvis == [1, 2, 3]


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".get_phasecentre"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".generate_lsm_from_csv"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".predict_vis"
)
def test_should_predict_visibilities_using_csv_lsm(
    predict_vis_mock, generate_lsm_from_csv_mock, get_phasecentre_mock
):

    upstream_output = UpstreamOutput()
    vis = Mock(name="Visibilities")
    vis.assign.return_value = [1, 2, 3]
    upstream_output["vis"] = vis
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_from_csv_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": False,
        "eb_ms": "test.ms",
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": None,
        "lsm_csv_path": "/path/to/lsm.csv",
        "fov": 10.0,
        "flux_limit": 1.0,
        "export_model_vis": False,
        "alpha0": -0.78,
        "reset_vis": False,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    get_phasecentre_mock.assert_called_once_with("path/to/input/ms")
    generate_lsm_from_csv_mock.assert_called_once_with(
        csvfile="/path/to/lsm.csv",
        phasecentre=(0.0, 0.0),
        fov=10.0,
        flux_limit=1.0,
    )
    predict_vis_mock.assert_called_once_with(
        upstream_output.vis.vis,
        upstream_output.vis.uvw,
        upstream_output.vis.datetime,
        upstream_output.vis.configuration,
        upstream_output.vis.antenna1,
        upstream_output.vis.antenna2,
        ["source1", "source2"],
        upstream_output.vis.phasecentre,
        beam_type="everybeam",
        eb_ms="test.ms",
        eb_coeffs="/path/to/coeffs",
    )

    assert result.modelvis == [1, 2, 3]


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".get_phasecentre"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".generate_lsm_from_csv"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".predict_vis"
)
def test_should_update_call_count(
    predict_vis_mock, generate_lsm_from_csv_mock, get_phasecentre_mock
):

    upstream_output = UpstreamOutput()
    vis = Mock(name="Visibilities")
    vis.assign.return_value = [1, 2, 3]
    upstream_output["vis"] = vis
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_from_csv_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": False,
        "eb_ms": "test.ms",
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": None,
        "lsm_csv_path": "/path/to/lsm.csv",
        "fov": 10.0,
        "flux_limit": 1.0,
        "export_model_vis": False,
        "alpha0": -0.78,
        "reset_vis": False,
    }

    upstream_output = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    upstream_output = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    assert upstream_output.get_call_count("predict_vis") == 2


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".get_phasecentre"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".predict_vis"
)
def test_should_throw_exception_if_lsm_is_none(
    predict_vis_mock, get_phasecentre_mock
):

    upstream_output = UpstreamOutput()
    vis = Mock(name="Visibilities")
    vis.assign.return_value = [1, 2, 3]
    upstream_output["vis"] = vis
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "normalise_at_beam_centre": False,
        "eb_ms": "test.ms",
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": None,
        "lsm_csv_path": None,
        "fov": 10.0,
        "flux_limit": 1.0,
        "export_model_vis": False,
        "alpha0": -0.78,
        "reset_vis": False,
    }

    with pytest.raises(RequiredArgumentMissingException):
        predict_vis_stage.stage_definition(
            upstream_output, **params, _cli_args_=cli_args
        )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".get_phasecentre"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".generate_lsm_from_gleamegc"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".predict_vis"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".prediction_central_beams"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".apply_gaintable_to_dataset"
)
def test_should_normalise_at_beam_centre(
    apply_gaintable_mock,
    prediction_beams_mock,
    predict_vis_mock,
    generate_lsm_mock,
    get_phasecentre_mock,
):
    vis = Mock(name="Visibilities")
    upstream_output = UpstreamOutput()
    vis = Mock(name="Visibilities")

    upstream_output["vis"] = vis
    cli_args = {"input": "path/to/input/ms"}

    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_mock.return_value = ["source1", "source2"]
    model_vis = Mock(name="Model Visibilities")
    vis.assign.return_value = model_vis
    predict_vis_mock.return_value = model_vis
    mock_beams = Mock(name="Beams")
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
        "fov": 10.0,
        "flux_limit": 1.0,
        "export_model_vis": False,
        "alpha0": -0.78,
        "reset_vis": False,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    prediction_beams_mock.assert_called_once_with(
        vis,
        beam_type="everybeam",
        eb_ms="path/to/input/ms",
        eb_coeffs="/path/to/coeffs",
    )

    apply_gaintable_mock.assert_has_calls(
        [
            call(vis, mock_beams, inverse=True),
            call(model_vis, mock_beams, inverse=True),
        ]
    )

    assert result.beams == mock_beams
    assert result.vis == normalised_vis
    assert result.modelvis == normalised_modelvis
