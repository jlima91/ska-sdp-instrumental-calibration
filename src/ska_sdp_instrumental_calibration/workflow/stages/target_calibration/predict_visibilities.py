from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from .._common import PREDICT_VISIBILITIES_COMMON_CONFIG
from ..model_visibilities import predict_visibilities


@ConfigurableStage(
    "predict_vis",
    configuration=Configuration(
        **PREDICT_VISIBILITIES_COMMON_CONFIG,
    ),
)
def predict_vis_stage(
    upstream_output,
    beam_type,
    eb_ms,
    eb_coeffs,
    gleamfile,
    lsm_csv_path,
    fov,
    flux_limit,
    alpha0,
    _cli_args_,
):
    return predict_visibilities(
        upstream_output=upstream_output,
        beam_type=beam_type,
        normalise_at_beam_centre=False,
        eb_ms=eb_ms,
        eb_coeffs=eb_coeffs,
        gleamfile=gleamfile,
        lsm_csv_path=lsm_csv_path,
        fov=fov,
        flux_limit=flux_limit,
        alpha0=alpha0,
        _cli_args_=_cli_args_,
    )
