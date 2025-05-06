# flake8: noqa: E501
import copy
import logging
import tempfile

import yaml
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.constants import DEFAULT_CLI_ARGS
from ska_sdp_piper.piper.pipeline import Pipeline
from ska_sdp_piper.piper.stage import Stages
from ska_sdp_piper.piper.utils.io_utils import read_yml, write_yml

from ska_sdp_instrumental_calibration.scheduler import DefaultScheduler
from ska_sdp_instrumental_calibration.workflow.stages import (
    bandpass_calibration_stage,
    delay_calibration_stage,
    export_gaintable_stage,
    generate_channel_rm_stage,
    load_data_stage,
    predict_vis_stage,
)

# from ska_sdp_instrumental_calibration.workflow.stages.delay_calibration import delay_calibration_stage

logger = logging.getLogger()

scheduler = DefaultScheduler()

# Create the pipeline instance
ska_sdp_instrumental_calibration = Pipeline(
    "ska_sdp_instrumental_calibration",
    stages=Stages(
        [
            load_data_stage,
            predict_vis_stage,
            bandpass_calibration_stage,
            generate_channel_rm_stage,
            delay_calibration_stage,
            export_gaintable_stage,
        ]
    ),
    scheduler=scheduler,
    global_config=Configuration(
        experimental=ConfigParam(
            dict,
            {"pipeline": []},
            description="""Configurations for experimental sub command.""",
        )
    ),
)


@ska_sdp_instrumental_calibration.sub_command(
    "experimental",
    DEFAULT_CLI_ARGS,
    help="Allows reordering of stages via additional config section",
)
def experimental(cli_args):
    """
    Reorder stages of INST pipeline. Use the config section
    global_parameters.experimental.stage_order to provide the order of
    callibration stages. Load data, predict and export stages are not
    reorder-able.

    Parameters
    ----------
        cli_args: argparse.Namespace
            CLI arguments
    """
    fixed_stages = [
        load_data_stage.name,
        export_gaintable_stage.name,
    ]

    stage_mapping = {
        stage.name: stage
        for stage in ska_sdp_instrumental_calibration._stages
        if stage.name not in fixed_stages
    }

    logger.warning("=========== INST Experimental ============")

    if cli_args.config_path:
        config = read_yml(cli_args.config_path)

        reconfigured_stages = []
        duplicate_counter = {}
        stage_order = (
            config.get("global_parameters", {})
            .get("experimental", {})
            .get("pipeline", [])
        )
        parameters = config.get("parameters", {})
        new_parameters = {}
        for stage_dict in stage_order:
            stage_name, stage_config = list(stage_dict.items())[0]

            if stage_name in fixed_stages:
                raise RuntimeError(
                    f"Mandatory stage {stage_name} included in the stage_order "
                    "section"
                )

            stage = stage_mapping[stage_name]

            if stage in reconfigured_stages:
                stage = copy.deepcopy(stage)
                duplicate_counter[stage_name] = (
                    duplicate_counter.get(stage_name, 0) + 1
                )
                new_stage_name = (
                    f"{stage_name}_{duplicate_counter[stage_name]}"
                )
                stage.name = new_stage_name

            if stage_config:
                new_parameters[stage.name] = stage_config
            elif stage_config := parameters.get(stage_name):
                new_parameters[stage.name] = stage_config

            reconfigured_stages.append(stage)

        if reconfigured_stages:
            stages = Stages(
                [
                    load_data_stage,
                    *reconfigured_stages,
                    export_gaintable_stage,
                ]
            )

            ska_sdp_instrumental_calibration._stages = stages
            config["parameters"] = new_parameters
            config["pipeline"] = {}
            _, temp_config = tempfile.mkstemp(text=True, suffix=".yml")
            write_yml(temp_config, config)
            cli_args.config_path = temp_config
            logger.info("Created temprory experimental config %s", temp_config)
        else:
            logger.warning(
                "No stage reordering provided. Using the default stage "
                "order"
            )
    else:
        logger.warning("No Config provided. Using the default stage order")
    logger.warning("==========================================")
    ska_sdp_instrumental_calibration._run(cli_args)
