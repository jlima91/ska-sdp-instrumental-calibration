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
    "instrumental_calibration.read_yml"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.write_yml"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.tempfile.mkstemp"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.Stages"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.copy.deepcopy"
)
def test_should_run_pipeline_with_custom_order_of_stages(
    deepcopy_mock,
    stages_mock,
    tempfile_mock,
    write_yml_mock,
    read_yml_mock,
    inst_pipeline_mock,
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = "path/to/config"

    mock_bandpass_1 = Mock(name="bandpass_mock_stage_1")
    mock_bandpass_2 = Mock(name="bandpass_mock_stage_2")
    mock_export_gaintable_1 = Mock(name="export_gaintable_mock_stage_1")
    mock_generate_rm_1 = Mock(name="mock_generate_rm_1")
    tempfile_mock.return_value = (1, "tmp/tempfile.yml")
    deepcopy_mock.side_effect = [
        mock_bandpass_1,
        mock_generate_rm_1,
        mock_bandpass_2,
        mock_export_gaintable_1,
    ]

    inst_pipeline_mock._stages = [
        load_data_stage,
        predict_vis_stage,
        bandpass_calibration_stage,
        generate_channel_rm_stage,
        export_gaintable_stage,
    ]

    read_yml_mock.return_value = {
        "global_parameters": {
            "experimental": {
                "pipeline": [
                    {"load_data": {}},
                    {"generate_channel_rm": {}},
                    {"bandpass_calibration": {}},
                    {"bandpass_calibration": {}},
                    {"export_gain_table": {}},
                    {"generate_channel_rm": {}},
                    {"bandpass_calibration": {}},
                    {"export_gain_table": {}},
                ]
            }
        }
    }
    stages_mock.return_value = stages_mock

    experimental(cli_args)

    deepcopy_mock.assert_has_calls(
        [
            call(bandpass_calibration_stage),
            call(generate_channel_rm_stage),
            call(bandpass_calibration_stage),
            call(export_gaintable_stage),
        ]
    )
    stages_mock.assert_called_once_with(
        [
            load_data_stage,
            generate_channel_rm_stage,
            bandpass_calibration_stage,
            mock_bandpass_1,
            export_gaintable_stage,
            mock_generate_rm_1,
            mock_bandpass_2,
            mock_export_gaintable_1,
        ]
    )
    assert mock_bandpass_1.name == "bandpass_calibration_1"
    assert mock_bandpass_2.name == "bandpass_calibration_2"
    assert mock_generate_rm_1.name == "generate_channel_rm_1"
    assert mock_export_gaintable_1.name == "export_gain_table_1"
    inst_pipeline_mock._run.assert_called_once_with(cli_args)


@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.ska_sdp_instrumental_calibration"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.read_yml"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.write_yml"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.tempfile.mkstemp"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.Stages"
)
def test_should_use_initial_config_provided(
    stages_mock,
    tempfile_mock,
    write_yml_mock,
    read_yml_mock,
    inst_pipeline_mock,
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = "path/to/config"
    tempfile_mock.return_value = (1, "tmp/tempfile.yml")

    inst_pipeline_mock._stages = [
        load_data_stage,
        predict_vis_stage,
        bandpass_calibration_stage,
        generate_channel_rm_stage,
        export_gaintable_stage,
    ]

    read_yml_mock.return_value = {
        "global_parameters": {
            "experimental": {
                "pipeline": [
                    {
                        "bandpass_calibration": {
                            "config_value": "updated_config"
                        }
                    },
                    {"bandpass_calibration": {}},
                ]
            }
        },
        "parameters": {
            "bandpass_calibration": {"config_value": "initial_config"}
        },
        "pipeline": {"export_gain_table": False},
    }
    stages_mock.return_value = stages_mock

    experimental(cli_args)
    read_yml_mock.assert_called_once_with("path/to/config")
    inst_pipeline_mock._run.assert_called_once_with(cli_args)
    tempfile_mock.assert_called_once_with(text=True, suffix=".yml")
    write_yml_mock.assert_called_once_with(
        "tmp/tempfile.yml",
        {
            "global_parameters": {
                "experimental": {
                    "pipeline": [
                        {
                            "bandpass_calibration": {
                                "config_value": "updated_config"
                            }
                        },
                        {"bandpass_calibration": {}},
                    ]
                }
            },
            "parameters": {
                "bandpass_calibration": {"config_value": "updated_config"},
                "bandpass_calibration_1": {"config_value": "initial_config"},
            },
            "pipeline": {},
        },
    )

    assert cli_args.config_path == "tmp/tempfile.yml"


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
    "instrumental_calibration.read_yml"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.pipelines."
    "instrumental_calibration.Stages"
)
def test_should_warn_user_if_no_stage_section_is_provided_in_experimental(
    stages_mock, read_yml_mock, logger_mock, inst_pipeline_mock
):

    cli_args = Mock(name="cli_args")
    cli_args.config_path = "/some/path"

    stages_mock.return_value = stages_mock

    read_yml_mock.return_value = {
        "global_parameters": {"experimental": {"pipeline": []}}
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
