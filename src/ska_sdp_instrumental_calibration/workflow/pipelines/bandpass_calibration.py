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

import dask.array as da
import numpy as np
from dask.distributed import Client, LocalCluster
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_func_python.preprocessing.flagger import rfi_flagger

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    load_ms,
    predict_vis,
    run_solver,
)
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.lsm_tmp import (
    generate_lsm,
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
        raise ValueError("GLEAM catalogue gleamegc.dat is required")
    eb_ms = pipeline_config.get("eb_ms", None)
    if eb_ms is None:
        raise ValueError("Name of Everybeam mock Measurement Set is required")
    eb_coeffs = pipeline_config.get("eb_coeffs", None)
    if eb_coeffs is None:
        raise ValueError("Path to Everybeam coeffs directory is required")

    # Filename
    ms_name = pipeline_config.get("ms_name", "demo.ms")
    hdf5_name = pipeline_config.get("hdf5_name", "demo.hdf5")

    # Sky model info
    fov = pipeline_config.get("fov_deg", 10)
    flux_limit = pipeline_config.get("flux_limit", 1)

    # Corruption and solver type
    gains = pipeline_config.get("gains", True)
    leakage = pipeline_config.get("leakage", False)
    solver = pipeline_config.get("solver", "gain_substitution")

    if ms_name == "demo.ms":
        # Generate a demo MSv2 Measurement Set
        logger.info(f"Generating a demo MSv2 Measurement Set {ms_name}")
        create_demo_ms(
            ms_name=ms_name,
            gains=gains,
            leakage=leakage,
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
    logger.info(f"Reading {ms_name} in {fchunk}-channel chunks")
    vis = load_ms(ms_name, fchunk)

    # Pre-processing
    #  - Is triggering the computation as is, so rfi_flagging=False for now.
    #  - Move to dask_wrappers? RFI flagging may need bandwidth...
    rfi_flagging = False
    if rfi_flagging:
        logger.info("Calling ska-sdp-func RFI flagger")
        vis = rfi_flagger(vis)

    # Get the LSM (single call for all channels / dask tasks)
    logger.info(f"Generating {gleamfile} LSM < {fov/2} deg > {flux_limit} Jy")
    lsm = generate_lsm(
        gleamfile=gleamfile,
        phasecentre=vis.phasecentre,
        fov=fov,
        flux_limit=flux_limit,
    )

    # Predict model visibilities
    logger.info(f"Predicting model visibilities in {fchunk}-channel chunks")
    modelvis = predict_vis(vis, lsm, eb_ms, eb_coeffs)

    # Call the solver
    logger.info(f"Running calibration in {fchunk}-channel chunks")
    gaintable = run_solver(
        vis=vis, modelvis=modelvis, solver=solver, niter=50, refant=0
    )

    # Output hdf5 file
    logger.info(f"Writing solutions to {hdf5_name}")
    gaintable.load()
    export_gaintable_to_hdf5([gaintable], hdf5_name)

    # Convergence checks (noise-free demo version)
    #  - Note that this runs the graph again. I tried assigning both vis
    #    and modelvis to gaintable for a single load, but it got confused
    #    by the baseline MultiIndex. MultiIndex causes a lot of trouble...
    #  - This is just a quick check, so it shouldn't hurt to run it again.
    if ms_name == "demo.ms":
        logger.info("Applying solutions")
        vis = apply_gaintable_to_dataset(vis, gaintable)
        logger.info("Checking results")
        if solver == "gain_substitution":
            # Don't expect convergence for these, so zeros them
            vis.vis.data[..., 1] = da.zeros(vis.vis.shape[:3], "complex")
            vis.vis.data[..., 2] = da.zeros(vis.vis.shape[:3], "complex")
            modelvis.vis.data[..., 1] = da.zeros(vis.vis.shape[:3], "complex")
            modelvis.vis.data[..., 2] = da.zeros(vis.vis.shape[:3], "complex")
        converged = np.allclose(modelvis.vis.data, vis.vis.data, atol=1e-6)
        if converged:
            logger.info("Convergence checks passed")
        else:
            logger.warning("Solving failed")

    # Shut down the scheduler and workers
    client.close()
    client.shutdown()
