# flake8: noqa: E501
import logging

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
)
