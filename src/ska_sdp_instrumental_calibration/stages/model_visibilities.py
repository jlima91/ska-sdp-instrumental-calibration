import logging

from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ..data_managers.beams import BeamsFactory
from ..data_managers.sky_model import GlobalSkyModel
from ..xarray_processors.apply import apply_gaintable_to_dataset
from ..xarray_processors.beams import prediction_central_beams
from ..xarray_processors.predict import predict_vis
from ._common import PREDICT_VISIBILITIES_COMMON_CONFIG

logger = logging.getLogger()


def predict_visibilities(
    upstream_output,
    beam_type,
    normalise_at_beam_centre,
    eb_ms,
    gleamfile,
    lsm_csv_path,
    export_sky_model,
    element_response_model,
    fov,
    flux_limit,
    alpha0,
    _output_dir_,
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
    gleamfile: str
        Path to the GLEAM catalog file.
    lsm_csv_path: str
        Specifies the location of CSV file containing the
        sky model. The CSV file should be in OSKAR CSV format.
    export_sky_model: bool
        Specifies whether to export the sky model to a CSV file.
    element_response_model: str
        type of element response model given to Everybeam.
        Defaulted oskar_dipole_cos.
        Refer documentation for more detials.
        https://everybeam.readthedocs.io/en/latest/tree/python/utils.html
    fov: float
        Field of view diameter in degrees for source selection
        (default: 10.0).
    flux_limit: float
        Minimum flux density in Jy for source selection
        (default: 1.0).
    alpha0: float
        Nominal alpha value to use when fitted
        data are unspecified. Default is -0.78.
    _output_dir_ : str
        Directory path where the output file will be written.
    _cli_args_: dict
        Command line arguments.

    Returns
    -------
    dict
        Updated upstream_output containing with modelvis.
    """
    upstream_output.add_checkpoint_key("modelvis")
    vis = upstream_output.vis
    gaintable = upstream_output.gaintable

    upstream_output["lsm"] = GlobalSkyModel(
        vis.phasecentre,
        fov,
        flux_limit,
        alpha0,
        gleamfile,
        lsm_csv_path,
    )

    if export_sky_model:
        sky_model_csv_path = f"{_output_dir_}/sky_model.csv"
        logger.info(f"Exporting sky model to CSV file at {sky_model_csv_path}")
        upstream_output["lsm"].export_sky_model_csv(sky_model_csv_path)

    beams_factory = None

    # Process beam related parameters
    eb_ms = _cli_args_["input"] if eb_ms is None else eb_ms

    if beam_type == "everybeam":
        logger.info("Using EveryBeam model in predict")

        beams_factory = BeamsFactory(
            nstations=vis.configuration.id.size,
            array_location=vis.configuration.location,
            direction=vis.phasecentre,
            ms_path=eb_ms,
            element_response_model=element_response_model,
        )

    modelvis = predict_vis(
        vis,
        upstream_output["lsm"],
        gaintable.time.data,
        gaintable.soln_interval_slices,
        beams_factory,
    )

    if normalise_at_beam_centre and beam_type == "everybeam":
        central_beams = prediction_central_beams(
            gaintable,
            beams_factory,
        )
        vis = apply_gaintable_to_dataset(vis, central_beams, inverse=True)
        modelvis = apply_gaintable_to_dataset(
            modelvis, central_beams, inverse=True
        )
        upstream_output["central_beams"] = central_beams
        upstream_output.add_checkpoint_key("central_beams")
        upstream_output["vis"] = vis

    upstream_output["modelvis"] = modelvis
    upstream_output["beams_factory"] = beams_factory
    upstream_output.increment_call_count("predict_vis")

    return upstream_output


predict_vis_stage = ConfigurableStage(
    "predict_vis",
    configuration=Configuration(
        **PREDICT_VISIBILITIES_COMMON_CONFIG,
    ),
)(predict_visibilities)
