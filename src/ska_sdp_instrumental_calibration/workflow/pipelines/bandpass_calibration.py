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
from astropy.coordinates import SkyCoord
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_func_python.preprocessing.flagger import rfi_flagger

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    load_ms,
    predict_vis,
    run_solver,
)
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.lsm import generate_lsm
from ska_sdp_instrumental_calibration.workflow.utils import create_demo_ms

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = setup_logger("pipeline.bandpass_calibration")


def run(pipeline_config) -> None:
    """Pipeline to generate bandpass calibration solutions.

    Args:
        pipeline_config (dict): Dictionary of configuration parameters.
            Must include gleamfile, eb_ms and eb_coeffs.
    Returns:
        None
    """

    # Filenames
    ms_name = pipeline_config.get("ms_name", "demo.ms")
    hdf5_name = pipeline_config.get("hdf5_name", "demo.hdf5")

    # Sky model info
    lsm = pipeline_config.get("lsm", None)
    fov = pipeline_config.get("fov_deg", 10)
    flux_limit = pipeline_config.get("flux_limit", 1)
    gleamfile = pipeline_config.get("gleamfile", None)
    # Check required external data
    if lsm is None and gleamfile is None:
        raise ValueError("Either a LSM or a catalogue file is required")

    # Beam model info
    beam_type = pipeline_config.get("beam_type", "everybeam")
    eb_coeffs = pipeline_config.get("eb_coeffs", None)
    eb_ms = pipeline_config.get("eb_ms", ms_name)
    if beam_type.lower() == "everybeam":
        # Required external data
        if eb_coeffs is None:
            raise ValueError("Path to Everybeam coeffs directory is required")
        logger.info(f"Initialising the EveryBeam telescope model with {eb_ms}")
    elif beam_type.lower() == "none":
        logger.info("Predicting model visibilities without a beam")
    else:
        raise ValueError(f"Unknown beam type: {beam_type}")

    # Corruption and solver type
    gains = pipeline_config.get("gains", True)
    leakage = pipeline_config.get("leakage", False)
    rotation = pipeline_config.get("rotation", False)
    solver = pipeline_config.get("solver", "gain_substitution")
    refant = 0

    if ms_name == "demo.ms":
        # Generate a demo MSv2 Measurement Set
        ntimes = pipeline_config.get("ntimes", 1)
        nchannels = pipeline_config.get("nchannels", 64)
        phasecentre = SkyCoord(ra=0.0, dec=-27.0, unit="degree")
        if lsm is None:
            # Get the LSM (single call for all channels / dask tasks)
            logger.info(f"LSM: {gleamfile} < {fov/2} deg > {flux_limit} Jy")
            lsm = generate_lsm(
                gleamfile=gleamfile,
                phasecentre=phasecentre,
                fov=fov,
                flux_limit=flux_limit,
            )
            logger.info(f"LSM: found {len(lsm)} components")

        logger.info(f"Generating a demo MSv2 Measurement Set {ms_name}")
        truetable = create_demo_ms(
            ms_name=ms_name,
            ntimes=ntimes,
            nchannels=nchannels,
            gains=gains,
            leakage=leakage,
            rotation=rotation,
            phasecentre=phasecentre,
            lsm=lsm,
            beam_type=beam_type,
            eb_coeffs=eb_coeffs,
            eb_ms=eb_ms,
        )

    # Dask info
    cluster = pipeline_config.get("dask_cluster", None)
    # The number of channels per frequency chunk
    fchunk = pipeline_config.get("fchunk", 16)

    logger.info(f"Starting pipeline with {fchunk}-channel chunks")

    # Set up a local dask cluster and client
    if cluster is None:
        logger.info("No dask cluster supplied. Using LocalCluster")
        cluster = LocalCluster()
    else:
        logger.info("Using existing dask cluster")
    client = Client(cluster)

    # Read in the Visibility dataset
    logger.info(f"Setting input from {ms_name} in {fchunk}-channel chunks")
    vis = load_ms(ms_name, fchunk)

    # Pre-processing
    #  - Is triggering the computation as is, so rfi_flagging=False for now.
    #  - Move to dask_wrappers? RFI flagging may need bandwidth...
    rfi_flagging = False
    if rfi_flagging:
        logger.info("Setting the ska-sdp-func RFI flagger")
        vis = rfi_flagger(vis)

    # Set up the local sky model (single call for all channels)
    #  - Could do this earlier, but currently use vis for phase centre
    if lsm is None:
        logger.info(f"LSM: {gleamfile} < {fov/2} deg > {flux_limit} Jy")
        lsm = generate_lsm(
            gleamfile=gleamfile,
            phasecentre=vis.phasecentre,
            fov=fov,
            flux_limit=flux_limit,
        )
        logger.info(f"LSM: found {len(lsm)} components")

    # Predict model visibilities
    logger.info(f"Setting vis predict in {fchunk}-channel chunks")
    modelvis = predict_vis(
        vis, lsm, beam_type=beam_type, eb_ms=eb_ms, eb_coeffs=eb_coeffs
    )

    # Call the solver
    logger.info(f"Setting calibration in {fchunk}-channel chunks")
    gaintable = run_solver(
        vis=vis, modelvis=modelvis, solver=solver, niter=50, refant=refant
    )

    # Output hdf5 file
    logger.info("Running graph and returning calibration solutions")
    gaintable.load()
    logger.info(f"Writing solutions to {hdf5_name}")
    export_gaintable_to_hdf5([gaintable], hdf5_name)

    # Convergence checks (noise-free demo version)
    if ms_name == "demo.ms":
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
        if solver == "gain_substitution":
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
    client.shutdown()
