from ska_sdp_piper.piper.v2.stage import ConfigurableStage

from ..model_visibilities import predict_visibilities

predict_vis_stage = ConfigurableStage(name="predict_vis")(predict_visibilities)
