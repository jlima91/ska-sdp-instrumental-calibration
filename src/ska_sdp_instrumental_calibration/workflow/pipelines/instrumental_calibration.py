# flake8: noqa: E501
import logging

import yaml
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.constants import DEFAULT_CLI_ARGS
from ska_sdp_piper.piper.pipeline import Pipeline
from ska_sdp_piper.piper.stage import Stages

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
            {"stage_order": []},
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
        predict_vis_stage.name,
        export_gaintable_stage.name,
    ]

    stage_mapping = {
        stage.name: stage
        for stage in ska_sdp_instrumental_calibration._stages
        if stage.name not in fixed_stages
    }

    logger.warning("=========== INST Experimental ============")

    if cli_args.config_path:
        with open(cli_args.config_path, "r") as f:
            config = yaml.safe_load(f)

            unique_stages = []
            stage_order = (
                config.get("global_parameters", {})
                .get("experimental", {})
                .get("stage_order", [])
            )
            for stage_name in stage_order:
                if stage_name in fixed_stages:
                    raise RuntimeError(
                        f"Mandatory stage {stage_name} included in the stage_order "
                        "section"
                    )

                if stage_name in unique_stages:
                    raise RuntimeError(
                        f"Duplicate stage {stage_name} in stage_order section"
                    )
                unique_stages.append(stage_name)

            if unique_stages:
                stages = Stages(
                    [
                        load_data_stage,
                        predict_vis_stage,
                        *[stage_mapping[stage] for stage in unique_stages],
                        export_gaintable_stage,
                    ]
                )

                ska_sdp_instrumental_calibration._stages = stages
            else:
                logger.warning(
                    "No stage reordering provided. Using the default stage "
                    "order"
                )
    else:
        logger.warning("No Config provided. Using the default stage order")
    logger.warning("==========================================")
    ska_sdp_instrumental_calibration._run(cli_args)
