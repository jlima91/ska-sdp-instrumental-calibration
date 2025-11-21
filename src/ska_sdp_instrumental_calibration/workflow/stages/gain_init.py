from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ._common import BANDPASS_COMMON_CONFIG
from .bandpass_calibration import bandpass_calibration

gain_init_stage = ConfigurableStage(
    "gain_init", configuration=Configuration(**BANDPASS_COMMON_CONFIG)
)(bandpass_calibration)
