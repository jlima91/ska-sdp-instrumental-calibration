"""Pipeline generate bandpass calibration solutions.

1. Generate demo MSv2 Measurement Set
2. Read the Measurement Set in frequency chunks
3. Initialisations
    - Get a Local Sky Model
    - RFI flagging
4. Predict model visibilities
    - Convert LSM to Skycomponents list
    - Convert list to vis dataset using dft_skycomponent_visibility
    - Apply Gaussian tapers to extended components
    - Apply everybeam beam models, per component, frequency and baseline
5. Solve for antenna-based gain terms and add to a GainTable dataset
6. Save GainTable dataset to HDF5
"""

import warnings

import numpy as np
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_func_python.preprocessing.flagger import rfi_flagger

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    ingest_predict_and_solve,
    load_ms,
    predict_vis,
    run_solver,
)
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.lsm import (
    generate_lsm_from_csv,
    generate_lsm_from_gleamegc,
)
from ska_sdp_instrumental_calibration.workflow.pipeline_config import (
    PipelineConfig,
)
from ska_sdp_instrumental_calibration.workflow.utils import get_phasecentre

from ...data_managers.data_export.export_to_h5parm import (
    export_gaintable_to_h5parm,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = setup_logger("pipeline.bandpass_calibration")


def run(pipeline_config) -> None:
    """Pipeline to generate bandpass calibration solutions.

    Args:
        pipeline_config (dict): Dictionary of configuration parameters.
    Returns:
        None
    """

    config = PipelineConfig(pipeline_config)

    if config.do_simulation:
        truetable = config.simulate_input_dataset()

    logger.info(f"Starting pipeline with {config.fchunk}-channel chunks")

    # Set up a local dask cluster and client
    if config.dask_scheduler_address is None:
        logger.info("No dask cluster supplied. Using LocalCluster")
        client = Client(LocalCluster())
    else:
        logger.info(
            f"Using existing dask cluster {config.dask_scheduler_address}"
        )
        client = Client(config.dask_scheduler_address)

    # Set up the local sky model (single call for all channels)
    if config.lsm is None:
        logger.info("Generating LSM for predict with:")
        logger.info(f" - Search radius: {config.fov/2} deg")
        logger.info(f" - Flux limit: {config.flux_limit} Jy")
        if config.gleamfile is not None:
            logger.info(f" - GLEAMEGC catalogue file: {config.gleamfile}")
            config.lsm = generate_lsm_from_gleamegc(
                gleamfile=config.gleamfile,
                phasecentre=get_phasecentre(config.ms_name),
                fov=config.fov,
                flux_limit=config.flux_limit,
            )
        elif config.csvfile is not None:
            logger.info(f" - csv file: {config.csvfile}")
            config.lsm = generate_lsm_from_csv(
                csvfile=config.csvfile,
                phasecentre=get_phasecentre(config.ms_name),
                fov=config.fov,
                flux_limit=config.flux_limit,
            )
        else:
            raise ValueError("Unknown sky model")
    logger.info(f"LSM contains {len(config.lsm)} components")

    if config.end_to_end_subbands:

        # Call the solver
        refant = 0
        logger.info(f"Setting calibration in {config.fchunk}-channel chunks")
        logger.info("end_to_end_subbands = true")
        gaintable = ingest_predict_and_solve(
            ms_name=config.ms_name,
            fchunk=config.fchunk,
            lsm=config.lsm,
            beam_type=config.beam_type,
            eb_ms=config.eb_ms,
            eb_coeffs=config.eb_coeffs,
            solver=config.solver,
            niter=50,
            refant=refant,
        )

    else:

        # Read in the Visibility dataset
        logger.info(
            f"Processing {config.ms_name} with {config.fchunk}-channel chunks"
        )
        vis = load_ms(config.ms_name, config.fchunk)

        # Pre-processing
        #  - Full-band operations trigger the computation, so leave for now.
        #  - Move to dask_wrappers?
        rfi_flagging = False
        if rfi_flagging:
            logger.info("Setting the ska-sdp-func RFI flagger")
            vis = rfi_flagger(vis)

        # Predict model visibilities
        logger.info(f"Setting vis predict in {config.fchunk}-channel chunks")
        modelvis = predict_vis(
            vis,
            config.lsm,
            beam_type=config.beam_type,
            eb_ms=config.eb_ms,
            eb_coeffs=config.eb_coeffs,
        )

        # Call the solver
        refant = 0
        logger.info(f"Setting calibration in {config.fchunk}-channel chunks")
        logger.info("end_to_end_subbands = false")
        gaintable = run_solver(
            vis=vis,
            modelvis=modelvis,
            solver=config.solver,
            niter=50,
            refant=refant,
        )

    # Output hdf5 file
    logger.info("Running graph and returning calibration solutions")
    gaintable.load()
    if config.h5parm_name is not None:
        logger.info(f"Writing solutions to {config.h5parm_name}")
        export_gaintable_to_h5parm(gaintable, config.h5parm_name)
    if config.hdf5_name is not None:
        logger.info(f"Writing solutions to {config.hdf5_name}")
        export_gaintable_to_hdf5(gaintable, config.hdf5_name)

    # Convergence checks (noise-free demo version)
    if config.do_simulation:
        logger.info("Checking results")
        gfit = gaintable.gain.data
        true = truetable.gain.data
        # Reference all polarisations again the X gain for the ref antenna
        gfit *= np.exp(
            -1j
            * np.angle(gfit[:, [refant], :, 0, 0][..., np.newaxis, np.newaxis])
        )
        true *= np.exp(
            -1j
            * np.angle(true[:, [refant], :, 0, 0][..., np.newaxis, np.newaxis])
        )
        if config.solver == "gain_substitution":
            # For independent Y calibration, reference Y gains separately
            gfit[:, :, :, 1, 1] *= np.exp(
                -1j * np.angle(gfit[:, [refant], :, 1, 1])
            )
            true[:, :, :, 1, 1] *= np.exp(
                -1j * np.angle(true[:, [refant], :, 1, 1])
            )
            # Fit won't change off-diag, so zero those before comparisons
            true[..., 0, 1] = true[..., 1, 0] = 0
        converged = np.allclose(gfit, true, atol=1e-6)
        if converged:
            logger.info("Convergence checks passed")
        else:
            logger.warning("Solving failed")

    # Shut down the scheduler and workers
    client.close()
    if config.dask_scheduler_address is None:
        client.shutdown()
