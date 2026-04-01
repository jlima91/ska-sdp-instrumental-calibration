from ska_sdp_piper.piper import ConfigurableStage

from ..predict_visibilities import predict_visibilities

predict_vis_stage = ConfigurableStage(name="predict_vis")(predict_visibilities)
