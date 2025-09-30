from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from .._common import PREDICT_VISIBILITIES_COMMON_CONFIG
from ..model_visibilities import predict_visibilities

predict_vis_stage = ConfigurableStage(
    "predict_vis",
    configuration=Configuration(
        **PREDICT_VISIBILITIES_COMMON_CONFIG,
    ),
)(predict_visibilities)
