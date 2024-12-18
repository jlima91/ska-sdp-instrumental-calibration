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
    load_ms,
    predict_vis,
    run_solver,
)
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.lsm import (
    generate_lsm_from_gleamegc,
)
from ska_sdp_instrumental_calibration.processing_tasks.post_processing import (
    model_rotations,
)
from ska_sdp_instrumental_calibration.workflow.pipeline_config import (
    PipelineConfig,
)

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

    # Read in the Visibility dataset
    logger.info(
        f"Will read from {config.ms_name} in {config.fchunk}-channel chunks"
    )
    vis = load_ms(config.ms_name, config.fchunk)

    # Pre-processing
    rfi_flagging = False  # not yet set up for dask chunks
    preproc_ave_time = 1  # not yet set up for dask chunks
    preproc_ave_frequency = 1  # not yet set up for dask chunks

    # Pre-processing
    #  - Is triggering the computation as is, so leave for now.
    #  - Move to dask_wrappers? RFI flagging may need bandwidth...
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
    if config.lsm is None:
        logger.info("Generating LSM for predict with:")
        logger.info(f" - Catalogue file: {config.gleamfile}")
        logger.info(f" - Search radius: {config.fov/2} deg")
        logger.info(f" - Flux limit: {config.flux_limit} Jy")
        config.lsm = generate_lsm_from_gleamegc(
            gleamfile=config.gleamfile,
            phasecentre=vis.phasecentre,
            fov=config.fov,
            flux_limit=config.flux_limit,
        )
        logger.info(f"LSM: found {len(config.lsm)} components")

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
    logger.info(" - Using solver jones_substitution")
    initialtable = run_solver(
        vis=vis,
        modelvis=modelvis,
        solver="jones_substitution",
        niter=20,
        refant=refant,
    )

    # Load all of the solutions into a numpy array
    logger.info("Running graph and returning calibration solutions")
    initialtable.load()

    # Fit for any differential rotations (single call for all channels).
    # Return gaintable filled with pure rotations.
    #  - If the vis data and model can fit into memory, it would be a good
    #    time to make them persistent. Otherwise they will be re-loaded and
    #    re-predicted in the next graph below.
    #  - Alternatively, export them to disk in a chunked way (e.g. to zarr)
    #  - Alternatively, can regenerate. Do this for now.
    logger.info("Fitting differential rotations")
    gaintable = model_rotations(initialtable, plot_sample=True).chunk(
        {"frequency": config.fchunk}
    )

    # Call the solver with updated initial solutions
    logger.info(f"Resetting calibration in {config.fchunk}-channel chunks")
    logger.info(" - Using solver normal_equations")
    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=gaintable,
        solver="normal_equations",
        niter=50,
        refant=refant,
    )

    # Output hdf5 file
    logger.info("Running graph and returning calibration solutions")
    gaintable.load()
    logger.info(f"Writing solutions to {config.hdf5_name}")
    export_gaintable_to_hdf5([gaintable], config.hdf5_name)

    # Convergence checks (noise-free demo version)
    #  - Note that this runs the graph again. I tried assigning both vis
    #    and modelvis to gaintable for a single load, but it got confused
    #    by the baseline MultiIndex. MultiIndex causes a lot of trouble...
    #  - This is just a quick check, so it shouldn't hurt to run it again.
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
        converged = np.allclose(gfit, true, atol=1e-6)
        if converged:
            logger.info("Convergence checks passed")
        else:
            logger.warning("Solving failed")

    # Shut down the scheduler and workers
    client.close()
    if config.dask_scheduler_address is None:
        client.shutdown()
