import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    predict_vis,
    prediction_central_beams,
)
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
        normalise_at_beam_centre=ConfigParam(
            bool,
            False,
            description="""If true, before running calibration, multiply vis
            and model vis by the inverse of the beam response in the
            beam pointing direction.""",
        ),
        eb_ms=ConfigParam(
            str,
            None,
            description="""If beam_type is "everybeam" but input ms does
            not have all of the metadata required by everybeam, this parameter
            is used to specify a separate dataset to use when setting up
            the beam models.""",
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
            description="""Specifies the location of CSV file containing the
            sky model. The CSV file should be in OSKAR CSV format.""",
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
    ),
)
def predict_vis_stage(
    upstream_output,
    beam_type,
    normalise_at_beam_centre,
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
    normalise_at_beam_centre: bool
        If true, before running calibration, multiply vis and model vis by
        the inverse of the beam response in the beam pointing direction
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
    upstream_output.add_checkpoint_key("modelvis")
    vis = upstream_output.vis

    logger.info("Generating LSM for predict with:")
    logger.info(f" - Search radius: {fov/2} deg")
    logger.info(f" - Flux limit: {flux_limit} Jy")

    phase_centre = get_phasecentre(_cli_args_["input"])

    if gleamfile is not None and lsm_csv_path is not None:
        logger.warning("LSM: GLEAMFILE and CSV provided. Using GLEAMFILE")

    if gleamfile is not None:
        logger.info(f" - Catalogue file: {gleamfile}")
        lsm = generate_lsm_from_gleamegc(
            gleamfile=gleamfile,
            phasecentre=phase_centre,
            fov=fov,
            flux_limit=flux_limit,
            alpha0=alpha0,
        )
    elif lsm_csv_path is not None:
        logger.info(f" - Catalogue file: {lsm_csv_path}")
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
    upstream_output["lsm"] = lsm
    upstream_output["beam_type"] = beam_type
    upstream_output["eb_coeffs"] = eb_coeffs

    logger.info(f"LSM: found {len(lsm)} components")

    eb_ms = _cli_args_["input"] if eb_ms is None else eb_ms
    upstream_output["eb_ms"] = eb_ms

    modelvis = predict_vis(
        vis,
        lsm,
        beam_type=beam_type,
        eb_ms=eb_ms,
        eb_coeffs=eb_coeffs,
    )

    if normalise_at_beam_centre:
        beams = prediction_central_beams(
            vis,
            beam_type=beam_type,
            eb_ms=eb_ms,
            eb_coeffs=eb_coeffs,
        ).persist()
        vis = apply_gaintable_to_dataset(vis, beams, inverse=True)
        modelvis = apply_gaintable_to_dataset(modelvis, beams, inverse=True)
        upstream_output["beams"] = beams
        upstream_output["vis"] = vis

    upstream_output["modelvis"] = modelvis
    upstream_output.increment_call_count("predict_vis")

    return upstream_output
