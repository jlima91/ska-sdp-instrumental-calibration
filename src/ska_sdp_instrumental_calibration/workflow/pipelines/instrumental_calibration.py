# flake8: noqa: E501
import logging

from ska_sdp_piper.piper.pipeline import Pipeline
from ska_sdp_piper.piper.stage import Stages

from ska_sdp_instrumental_calibration.scheduler import DefaultScheduler
from ska_sdp_instrumental_calibration.workflow.stages.load_data import (
    load_data,
)
from ska_sdp_instrumental_calibration.workflow.stages.stage_2 import stage_2
from ska_sdp_instrumental_calibration.workflow.stages.stage_3 import stage_3

logger = logging.getLogger()

scheduler = DefaultScheduler()

# Create the pipeline instance
ska_sdp_instrumental_calibration = Pipeline(
    "ska_sdp_instrumental_calibration",
    stages=Stages(
        [
            load_data,
            stage_2,
            stage_3,
        ]
    ),
    scheduler=scheduler,
)
