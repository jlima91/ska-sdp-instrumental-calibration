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
from ska_sdp_func_python.preprocessing.averaging import (
    averaging_frequency,
    averaging_time,
)
from ska_sdp_func_python.preprocessing.flagger import rfi_flagger

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    calibrate_dataset,
    load_ms,
    predict_vis,
    run_solver,
)
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.lsm_tmp import (
    generate_lsm,
)
from ska_sdp_instrumental_calibration.processing_tasks.post_processing import (
    model_rotations,
)
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

    # Required external data
    gleamfile = pipeline_config.get("gleamfile", None)
    if gleamfile is None:
        raise ValueError("GLEAM catalogue gleamegc.dat is required.")
    eb_ms = pipeline_config.get("eb_ms", None)
    if eb_ms is None:
        raise ValueError("Name of Everybeam mock Measurement Set is required.")
    eb_coeffs = pipeline_config.get("eb_coeffs", None)
    if eb_coeffs is None:
        raise ValueError("Path to Everybeam coeffs directory is required.")

    # Filename
    ms_name = pipeline_config.get("ms_name", "demo.ms")
    hdf5_name = pipeline_config.get("hdf5_name", "demo.hdf5")

    # Sky model info
    fov = pipeline_config.get("fov_deg", 10)
    flux_limit = pipeline_config.get("flux_limit", 1)

    # Pre-processing
    rfi_flagging = False  # not yet set up for dask chunk
    preproc_ave_time = 1  # not yet set up for dask chunk
    preproc_ave_frequency = 1  # not yet set up for dask chunk

    if ms_name == "demo.ms":
        # Generate a demo MSv2 Measurement Set
        logger.info(f"Generating a demo MSv2 Measurement Set {ms_name}.")
        create_demo_ms(
            ms_name=ms_name,
            gains=True,
            leakage=True,
            rotation=True,
            wide_channels=True,
            gleamfile=gleamfile,
            eb_ms=eb_ms,
            eb_coeffs=eb_coeffs,
        )

    # Set up a local dask cluster and client
    cluster = LocalCluster()
    client = Client(cluster)

    # Set the number of channels per frequency chunk
    fchunk = 16

    # Read in the Visibility dataset
    logger.info(f"Reading {ms_name} in {fchunk}-channel chunks.")
    vis = load_ms(client, ms_name, fchunk)

    # Pre-processing
    #  - Move these to dask_wrappers.
    #  - Do RFI flagging?
    if rfi_flagging:
        logger.info("Calling rfi_flagger")
        vis = rfi_flagger(vis)
    #  - Time averaging?
    if preproc_ave_time > 1:
        logger.info(f"Averaging dataset by {preproc_ave_time} time steps")
        vis = averaging_time(vis, timestep=preproc_ave_time)
    #  - Frequency averaging?
    if preproc_ave_frequency > 1:
        logger.info(f"Averaging dataset by {preproc_ave_frequency} channels")
        # Auto average to 781.25 kHz?
        # dfrequency_bf = 781.25e3
        # dfrequency = vis.frequency.data[1] - vis.frequency.data[0]
        # freqstep = int(numpy.round(dfrequency_bf / dfrequency))
        vis = averaging_frequency(vis, freqstep=preproc_ave_frequency)

    # Get the LSM (single call for all channels)
    logger.info(f"Generating {gleamfile} LSM < {fov/2} deg > {flux_limit} Jy.")
    lsm = generate_lsm(
        gleamfile=gleamfile,
        phasecentre=vis.phasecentre,
        fov=fov,
        flux_limit=flux_limit,
    )

    # Predict model visibilities
    logger.info(f"Predicting model visibilities in {fchunk}-channel chunks.")
    modelvis = predict_vis(client, vis, lsm, eb_ms, eb_coeffs)

    # Call the solver
    logger.info(f"Running calibration in {fchunk}-channel chunks.")
    initialtable = run_solver(
        client,
        vis=vis,
        modelvis=modelvis,
        solver="jones_substitution",
        niter=20,
        refant=0,
    )

    # Load all of the solutions into a numpy array and fit for any
    # differential rotations (single call for all channels).
    # Return gaintable filled with pure rotations.
    initialtable.load()
    gaintable = model_rotations(initialtable, plot_sample=True).chunk(
        {"frequency": fchunk}
    )

    # Call the solver with updated initial solutions
    logger.info(f"Rerunning calibration in {fchunk}-channel chunks.")
    gaintable = run_solver(
        client,
        vis=vis,
        modelvis=modelvis,
        gaintable=gaintable,
        solver="normal_equations",
        niter=50,
        refant=0,
    )

    # Output hdf5 file
    logger.info(f"Writing solutions to {hdf5_name}.")
    export_gaintable_to_hdf5([gaintable], hdf5_name)

    # Final checks (demo version)
    if ms_name == "demo.ms":

        logger.info("Applying solutions.")
        vis = calibrate_dataset(client, vis, modelvis, gaintable)

        logger.info("Checking results.")
        vis.load()
        modelvis.load()
        converged = np.allclose(modelvis.vis.data, vis.vis.data, atol=1e-6)
        if converged:
            logger.info("Solutions converged. Checks passed")
        else:
            logger.warning("Solutions did not converge.")

    # Shut down the scheduler and workers
    client.close()
    client.shutdown()
