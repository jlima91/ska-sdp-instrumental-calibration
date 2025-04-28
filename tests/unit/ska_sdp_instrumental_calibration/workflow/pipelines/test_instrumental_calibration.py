import pytest
from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.workflow.pipelines.instrumental_calibration import (  # noqa: E501
    experimental,
)
from ska_sdp_instrumental_calibration.workflow.stages import (
    bandpass_calibration_stage,
    export_gaintable_stage,
    generate_channel_rm_stage,
    load_data_stage,
    predict_vis_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.ska_sdp_instrumental_calibration"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.yaml.safe_load"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.Stages"
)
@patch("builtins.open")
def test_should_run_pipeline_with_custom_order_of_stages(
    open_mock, stages_mock, safe_load_mock, inst_pipeline_mock
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = "path/to/config"

    file_mock = Mock(name="file_mock")
    open_mock.__enter__.return_value = file_mock

    inst_pipeline_mock._stages = [
        load_data_stage,
        predict_vis_stage,
        bandpass_calibration_stage,
        generate_channel_rm_stage,
        export_gaintable_stage,
    ]

    safe_load_mock.return_value = {
        "global_parameters": {
            "experimental": {
                "stage_order": ["generate_channel_rm", "bandpass_calibration"]
            }
        }
    }
    stages_mock.return_value = stages_mock

    experimental(cli_args)

    stages_mock.assert_called_once_with(
        [
            load_data_stage,
            predict_vis_stage,
            generate_channel_rm_stage,
            bandpass_calibration_stage,
            export_gaintable_stage,
        ]
    )

    inst_pipeline_mock._run.assert_called_once_with(cli_args)


@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.yaml.safe_load"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.Stages"
)
@patch("builtins.open")
def test_should_validate_for_reorderable_stages(
    open_mock, stages_mock, safe_load_mock
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = "path/to/config"

    file_mock = Mock(name="file_mock")
    open_mock.__enter__.return_value = file_mock

    stages_mock.return_value = stages_mock

    safe_load_mock.return_value = {
        "global_parameters": {
            "experimental": {
                "stage_order": ["bandpass_calibration", "load_data"]
            }
        }
    }
    with pytest.raises(RuntimeError):
        experimental(cli_args)

    safe_load_mock.return_value = {
        "global_parameters": {
            "experimental": {
                "stage_order": ["bandpass_calibration", "bandpass_calibration"]
            }
        }
    }
    with pytest.raises(RuntimeError):
        experimental(cli_args)


@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.ska_sdp_instrumental_calibration"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.logger"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.yaml.safe_load"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.Stages"
)
@patch("builtins.open")
def test_should_warn_user_if_no_stage_section_is_provided_in_experimental(
    open_mock, stages_mock, safe_load_mock, logger_mock, inst_pipeline_mock
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = "/some/path"

    file_mock = Mock(name="file_mock")
    open_mock.__enter__.return_value = file_mock

    stages_mock.return_value = stages_mock

    safe_load_mock.return_value = {
        "global_parameters": {"experimental": {"stage_order": []}}
    }

    experimental(cli_args)

    logger_mock.warning.assert_has_calls(
        [call("No stage reordering provided. Using the default stage order")]
    )

    inst_pipeline_mock._run.assert_called_once_with(cli_args)


@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.ska_sdp_instrumental_calibration"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.logger"
)
def test_should_warn_user_if_no_config_is_provided_in_experimental(
    logger_mock, inst_pipeline_mock
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = None

    experimental(cli_args)

    logger_mock.warning.assert_has_calls(
        [call("No Config provided. Using the default stage order")]
    )

    inst_pipeline_mock._run.assert_called_once_with(cli_args)
