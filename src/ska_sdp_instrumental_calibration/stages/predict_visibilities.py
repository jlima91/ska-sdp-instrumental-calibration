import logging
from typing import Annotated, Optional

from pydantic import Field
from ska_sdp_piper.piper import CLIArgument, ConfigurableStage

from ..data_managers.beams import BeamsFactory
from ..data_managers.sky_model import GlobalSkyModel
from ..sdm import SDM
from ..xarray_processors.apply import apply_gaintable_to_dataset
from ..xarray_processors.beams import prediction_central_beams
from ..xarray_processors.predict import predict_vis

logger = logging.getLogger()


def predict_visibilities(
    _upstream_output_,
    _qa_dir_,
    input: Annotated[list[str], CLIArgument],
    sdm_path: Annotated[Optional[str], CLIArgument] = None,
    use_everybeam: Annotated[
        bool,
        Field(description="Whether to use everybeam model."),
    ] = True,
    normalise_at_beam_centre: Annotated[
        bool,
        Field(
            description="""If true, before running calibration, multiply vis
            and model vis by the inverse of the beam response in the
            beam pointing direction.""",
        ),
    ] = True,
    element_response_model: Annotated[
        str,
        Field(
            description="""Type of element response model.
            Required if use_everybeam is True.
            Refer documentation for more details:
            https://everybeam.readthedocs.io/en/latest/tree/python/utils.html
            """
        ),
    ] = "oskar_dipole_cos",
    eb_ms: Annotated[
        Optional[str],
        Field(
            description="""If everybeam is being used but input ms does
            not have all of the metadata required by everybeam, this parameter
            is used to specify a separate dataset to use when setting up
            the beam models."""
        ),
    ] = None,
    gleamfile: Annotated[
        Optional[str],
        Field(
            description="""Specifies the location of gleam catalogue
            file gleamegc.dat"""
        ),
    ] = None,
    lsm_csv_path: Annotated[
        Optional[str],
        Field(
            description="""Specifies the location of CSV file containing the
            sky model. The CSV file should be in OSKAR CSV format."""
        ),
    ] = None,
    sdm_lsm_file: Annotated[
        str,
        Field(
            description="""Specifies name of LSM file available in SDM which
            contains the local sky model. This will be used if sdm_path is
            provided."""
        ),
    ] = "sky_model.csv",
    export_sky_model: Annotated[
        bool,
        Field(
            description="""Specifies whether to export the sky model
            to a CSV file."""
        ),
    ] = False,
    fov: Annotated[
        float,
        Field(
            description="""Specifies the width of the cone used when
            searching for components, in units of degrees."""
        ),
    ] = 5.0,
    flux_limit: Annotated[
        float,
        Field(
            description="""Specifies the flux density limit used when
            searching for components, in units of Jy."""
        ),
    ] = 1.0,
    alpha0: Annotated[
        float,
        Field(
            description="""Nominal alpha value to use when fitted data
            are unspecified."""
        ),
    ] = -0.78,
):
    """
    Predict model visibilities using a local sky model.

    Parameters
    ----------
    _upstream_output_: dict
        Output from the upstream stage.
    _qa_dir_ : str
        Directory path where the diagnostic QA outputs will be written.
    input: CLIArgument
        Input measurementset.
    use_everybeam: bool
        Whether to use everybeam model. It uses everybeam by default.
    normalise_at_beam_centre: bool
        If true, before running calibration, multiply vis and model vis by
        the inverse of the beam response in the beam pointing direction.
    element_response_model: str
        type of element response model given to Everybeam.
        Defaulted oskar_dipole_cos.
        Refer documentation for more detials.
        https://everybeam.readthedocs.io/en/latest/tree/python/utils.html
    eb_ms: str
        If everybeam is being used but input ms does
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
    fov: float
        Field of view diameter in degrees for source selection
        (default: 10.0).
    flux_limit: float
        Minimum flux density in Jy for source selection
        (default: 1.0).
    alpha0: float
        Nominal alpha value to use when fitted
        data are unspecified. Default is -0.78.

    Returns
    -------
    dict
        Updated upstream_output containing with modelvis.
    """
    _upstream_output_.add_checkpoint_key("modelvis")
    vis = _upstream_output_.vis
    gaintable = _upstream_output_.gaintable
    if sdm_path is not None:
        lsm_csv_path = str(
            SDM.SKY.find_model(
                sdm_path, _upstream_output_.field_id, sdm_lsm_file
            )
        )
    _upstream_output_["lsm"] = GlobalSkyModel(
        vis.phasecentre,
        fov,
        flux_limit,
        alpha0,
        gleamfile,
        lsm_csv_path,
    )

    if export_sky_model:
        ms_prefix = _upstream_output_.ms_prefix
        sky_model_csv_path = f"{_qa_dir_}/{ms_prefix}_sky_model.csv"
        logger.info(f"Exporting sky model to CSV file at {sky_model_csv_path}")
        _upstream_output_["lsm"].export_sky_model_csv(sky_model_csv_path)

    beams_factory = None

    # Process beam related parameters

    if use_everybeam:
        logger.info("Using EveryBeam model in predict")
        eb_ms = input[0] if eb_ms is None else eb_ms

        beams_factory = BeamsFactory(
            nstations=vis.configuration.id.size,
            array_location=vis.configuration.location,
            direction=vis.phasecentre,
            ms_path=eb_ms,
            element_response_model=element_response_model,
        )

    modelvis = predict_vis(
        vis,
        _upstream_output_["lsm"],
        gaintable.time.data,
        gaintable.soln_interval_slices,
        beams_factory,
    )

    if normalise_at_beam_centre and use_everybeam:
        central_beams = prediction_central_beams(
            gaintable,
            beams_factory,
        )
        vis = apply_gaintable_to_dataset(vis, central_beams, inverse=True)
        modelvis = apply_gaintable_to_dataset(
            modelvis, central_beams, inverse=True
        )
        _upstream_output_["central_beams"] = central_beams
        _upstream_output_.add_checkpoint_key("central_beams")
        _upstream_output_["vis"] = vis

    _upstream_output_["modelvis"] = modelvis
    _upstream_output_["beams_factory"] = beams_factory
    _upstream_output_.increment_call_count("predict_vis")

    return _upstream_output_


predict_vis_stage = ConfigurableStage(name="predict_vis")(predict_visibilities)
