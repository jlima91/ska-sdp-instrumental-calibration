import pytest
from mock import Mock, patch

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
    upstream_output["vis"] = Mock(name="Visibilities")
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
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
        upstream_output.vis,
        ["source1", "source2"],
        beam_type="everybeam",
        eb_ms="path/to/input/ms",
        eb_coeffs="/path/to/coeffs",
        reset_vis=False,
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
    upstream_output["vis"] = Mock(name="Visibilities")
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_from_csv_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
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
        upstream_output.vis,
        ["source1", "source2"],
        beam_type="everybeam",
        eb_ms="test.ms",
        eb_coeffs="/path/to/coeffs",
        reset_vis=False,
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
    upstream_output["vis"] = Mock(name="Visibilities")
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_from_csv_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
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
    upstream_output["vis"] = Mock(name="Visibilities")
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
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
