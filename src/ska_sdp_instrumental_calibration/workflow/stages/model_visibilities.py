import logging

from ska_sdp_piper.piper.configurations import Configuration
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
from ._common import PREDICT_VISIBILITIES_COMMON_CONFIG

logger = logging.getLogger()


def predict_visibilities(
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

    phase_centre = vis.phasecentre

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
    # metadata = upstream_output["metadata"]
    # jones_eb = beam_model(
    #     vis=vis,
    #     metadata=metadata,
    # )
    # modelvis, vis = do_centre_correct(modelvis, vis, jones_eb, metadata)

    upstream_output["modelvis"] = modelvis
    upstream_output["vis"] = vis
    upstream_output.increment_call_count("predict_vis")

    return upstream_output


predict_vis_stage = ConfigurableStage(
    "predict_vis",
    configuration=Configuration(
        **PREDICT_VISIBILITIES_COMMON_CONFIG,
    ),
)(predict_visibilities)
