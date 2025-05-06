import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import predict_vis
from ...exceptions import RequiredArgumentMissingException
from ...processing_tasks.lsm import (
    generate_lsm_from_csv,
    generate_lsm_from_gleamegc,
)
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
        lsm_csv_path=ConfigParam(
            str,
            None,
            description="""Specifies the location of CSV file for custom
            components""",
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
        alpha0=ConfigParam(
            float,
            -0.78,
            description="""Nominal alpha value to use when fitted data
            are unspecified. Default is -0.78.""",
        ),
        reset_vis=ConfigParam(
            bool,
            False,
            description="""Whether or not to set visibilities to zero before
            accumulating components. Default is False.""",
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
    lsm_csv_path,
    fov,
    flux_limit,
    alpha0,
    reset_vis,
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
        lsm_csv_path : str
            Path to the Custom component CSV.
        fov : float
            Field of view diameter in degrees for source selection\
                  (default: 10.0).
        flux_limit : float
            Minimum flux density in Jy for source selection
            (default: 1.0).
        export_model_vis : bool
            Whether to export model visibilities (default: False).
        alpha0: float
            Nominal alpha value to use when fitted
            data are unspecified. Default is -0.78.
        reset_vis: bool
            Whether or not to set visibilities to zero before
            accumulating components. Default is False.
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

    phase_centre = get_phasecentre(_cli_args_["input"])

    if gleamfile is not None and lsm_csv_path is not None:
        logger.warning("LSM: GLEAMFILE and CSV provided. Using GLEAMFILE")

    if gleamfile is not None:
        lsm = generate_lsm_from_gleamegc(
            gleamfile=gleamfile,
            phasecentre=phase_centre,
            fov=fov,
            flux_limit=flux_limit,
            alpha0=alpha0,
        )
    elif lsm_csv_path is not None:
        lsm = generate_lsm_from_csv(
            csvfile=lsm_csv_path,
            phasecentre=phase_centre,
            fov=fov,
            flux_limit=flux_limit,
        )
    else:
        raise RequiredArgumentMissingException(
            "No LSM components provided. "
            "Either provide GLEAMFILE or LSM CSV file"
        )

    logger.info(f"LSM: found {len(lsm)} components")

    modelvis = predict_vis(
        vis,
        lsm,
        beam_type=beam_type,
        eb_ms=eb_ms,
        eb_coeffs=eb_coeffs,
        reset_vis=reset_vis,
    )

    if export_model_vis:
        # [TODO] : export the model visibilities to file.
        pass

    upstream_output["modelvis"] = modelvis

    return upstream_output
