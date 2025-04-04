import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import predict_vis
from ...processing_tasks.lsm import generate_lsm_from_gleamegc
from ..utils import get_phasecentre

logger = logging.getLogger()


@ConfigurableStage(
    "predict_vis",
    configuration=Configuration(
        beam_type=ConfigParam(
            str,
            "everybeam",
            description="Type of beam model to use. Default is 'everybeam'",
        ),
        eb_ms=ConfigParam(
            str,
            None,
            description="""Measurement set need to initialise the everybeam
            telescope. Required if bbeam_type is 'everybeam'.""",
        ),
        eb_coeffs=ConfigParam(
            str,
            None,
            description="""Everybeam coeffs datadir containing beam
            coefficients. Required if bbeam_type is 'everybeam'.""",
        ),
        gleamfile=ConfigParam(
            str,
            None,
            description="""Specifies the location of gleam catalogue
            file gleamegc.dat""",
        ),
        fov=ConfigParam(
            float,
            10.0,
            description="""Specifies the width of the cone used when
            searching for compoents, in units of degrees. Default: 10.""",
        ),
        flux_limit=ConfigParam(
            float,
            1.0,
            description="""Specifies the flux density limit used when
            searching for compoents, in units of Jy. Defaults to 1""",
        ),
        export_model_vis=ConfigParam(
            bool, False, "Export predicted model visibilities"
        ),
    ),
)
def predict_vis_stage(
    upstream_output,
    beam_type,
    eb_ms,
    eb_coeffs,
    gleamfile,
    fov,
    flux_limit,
    export_model_vis,
    _cli_args_,
):
    """
    Predict model visibilities using a local sky model.

    Parameters
    ----------
        upstream_output : dict
            Output from the upstream stage.
        beam_type : str
            Type of beam model to use (default: 'everybeam').
        eb_ms : str
            Path to measurement set for everybeam initialization.
            Required when beam_type is 'everybeam'.
        eb_coeffs : str
            Path to everybeam coefficients directory.
            Required when beam_type is 'everybeam'.
        gleamfile : str
            Path to the GLEAM catalog file.
        fov : float
            Field of view diameter in degrees for source selection\
                  (default: 10.0).
        flux_limit : float
            Minimum flux density in Jy for source selection (default: 1.0).
        export_model_vis : bool
            Whether to export model visibilities (default: False).
        _cli_args_ : dict
            Command line arguments.
    Returns
    -------
        dict
            Updated upstream_output containing with modelvis.
    """

    vis = upstream_output.vis

    logger.info("Generating LSM for predict with:")
    logger.info(f" - Catalogue file: {gleamfile}")
    logger.info(f" - Search radius: {fov/2} deg")
    logger.info(f" - Flux limit: {flux_limit} Jy")
    lsm = generate_lsm_from_gleamegc(
        gleamfile=gleamfile,
        phasecentre=get_phasecentre(_cli_args_["input"]),
        fov=fov,
        flux_limit=flux_limit,
    )
    logger.info(f"LSM: found {len(lsm)} components")

    modelvis = predict_vis(
        vis,
        lsm,
        beam_type=beam_type,
        eb_ms=eb_ms,
        eb_coeffs=eb_coeffs,
    )

    if export_model_vis:
        # [TODO] : export the model visibilities to file.
        pass

    upstream_output["modelvis"] = modelvis

    return upstream_output
