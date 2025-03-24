# flake8: noqa: E501
import logging

from ska_sdp_piper.piper.pipeline import Pipeline
from ska_sdp_piper.piper.stage import Stages

from ska_sdp_instrumental_calibration.scheduler import DefaultScheduler

# from ska_sdp_spectral_line_imaging.scheduler import DefaultScheduler
from ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration import (
    bandpass_calibration,
)
from ska_sdp_instrumental_calibration.workflow.stages.complex_gain import (
    complex_gain,
)
from ska_sdp_instrumental_calibration.workflow.stages.delay_calibration import (
    delay_calibration,
)
from ska_sdp_instrumental_calibration.workflow.stages.faraday_rotation import (
    faraday_rotation,
)
from ska_sdp_instrumental_calibration.workflow.stages.flag_ms import flag_ms
from ska_sdp_instrumental_calibration.workflow.stages.flux_calibration import (
    flux_calibration,
)
from ska_sdp_instrumental_calibration.workflow.stages.load_data import (
    load_data,
)

logger = logging.getLogger()

scheduler = DefaultScheduler()

# Create the pipeline instance
instrumental_calibration_pipeline = Pipeline(
    "instrumental_calibration_pipeline",
    stages=Stages(
        [
            load_data,
            flag_ms,
            delay_calibration,
            bandpass_calibration,
            flux_calibration,
            complex_gain,
            faraday_rotation,
        ]
    ),
    scheduler=scheduler,
)
