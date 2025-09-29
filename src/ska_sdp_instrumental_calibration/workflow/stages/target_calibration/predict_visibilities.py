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
    """
    Predict model visibilities using a local sky model.

    Parameters
    ----------
    upstream_output: dict
        Output from the upstream stage.
    beam_type: str
        Type of beam model to use (default: 'everybeam').
    eb_ms: str
        If beam_type is "everybeam" but input ms does
        not have all of the metadata required by everybeam, this parameter
        is used to specify a separate dataset to use when setting up
        the beam models.
    eb_coeffs: str
        Path to everybeam coefficients directory.
        Required when beam_type is 'everybeam'.
    gleamfile: str
        Path to the GLEAM catalog file.
    lsm_csv_path: str
        Specifies the location of CSV file containing the
        sky model. The CSV file should be in OSKAR CSV format.
    fov: float
        Field of view diameter in degrees for source selection
        (default: 10.0).
    flux_limit: float
        Minimum flux density in Jy for source selection
        (default: 1.0).
    alpha0: float
        Nominal alpha value to use when fitted
        data are unspecified. Default is -0.78.
    _cli_args_: dict
        Command line arguments.

    Returns
    -------
    dict
        Updated upstream_output containing with modelvis.
    """
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
