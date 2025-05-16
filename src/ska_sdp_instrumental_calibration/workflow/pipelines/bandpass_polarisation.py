"""Pipeline generate bandpass calibration solutions.

1. Generate demo MSv2 Measurement Set
2. Read the Measurement Set in frequency chunks
3. Initialisations
    - Get a Local Sky Model
    - Preprocessing (flagging and averaging) -- currently disabled
4. Predict model visibilities
    - Convert LSM to Skycomponents list
    - Convert list to vis dataset using dft_skycomponent_visibility
    - Apply Gaussian tapers to extended components
    - Apply everybeam beam models, per component, frequency and baseline
5. Solve for station- and frequency-based Jones matrices
6. Solve for differential Faraday rotation (DI RM per station)
7. Re-predict model visibilities
    - Apply station-based RM estimates to components before beam models
8. Re-solve for station- and frequency-based Jones matrices
9. Save GainTable dataset to H5Parm
"""

import warnings

import matplotlib.pyplot as plt
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
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.processing_tasks.lsm import (
    generate_lsm_from_csv,
    generate_lsm_from_gleamegc,
)
from ska_sdp_instrumental_calibration.processing_tasks.post_processing import (
    model_rotations,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    generate_rotation_matrices,
)
from ska_sdp_instrumental_calibration.workflow.pipeline_config import (
    PipelineConfig,
)
from ska_sdp_instrumental_calibration.workflow.utils import get_phasecentre

from ...data_managers.data_export.export_gaintable import (
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
        config.simulate_input_dataset()

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
        logger.warning("Not running end-to-end version.")

    # If the vis data and model can fit into memory, the load_ms and
    # predict_vis calls could be returned with .persist() so that they do
    # not need to be run again for the second run_solver call.
    # For now they will just be regenerated.

    # Read in the Visibility dataset
    logger.info(
        f"Will ingest {config.ms_name} in {config.fchunk}-channel chunks"
    )
    vis = load_ms(config.ms_name, config.fchunk)  # .persist()

    # Pre-processing
    rfi_flagging = False  # not yet set up for dask chunks
    preproc_ave_time = 1  # not yet set up for dask chunks
    preproc_ave_frequency = 1  # not yet set up for dask chunks

    # Pre-processing
    #  - Full-band operations trigger the computation, so leave for now.
    #  - Move to dask_wrappers?
    #     - Any time or freq averaging should perhaps be added to load_ms
    #  - Do RFI flagging?
    if rfi_flagging:
        logger.info("Calling rfi_flagger")
        vis = rfi_flagger(vis)
    #  - Time averaging?
    if preproc_ave_time > 1:
        logger.info(f"Averaging vis by {preproc_ave_time} time steps")
        vis = averaging_time(vis, timestep=preproc_ave_time)
    #  - Frequency averaging?
    if preproc_ave_frequency > 1:
        logger.info(f"Averaging vis by {preproc_ave_frequency} channels")
        # Auto average to 781.25 kHz?
        # dfrequency_bf = 781.25e3
        # dfrequency = vis.frequency.data[1] - vis.frequency.data[0]
        # freqstep = int(numpy.round(dfrequency_bf / dfrequency))
        vis = averaging_frequency(vis, freqstep=preproc_ave_frequency)

    # Predict model visibilities
    logger.info(f"Setting vis predict in {config.fchunk}-channel chunks")
    modelvis = predict_vis(
        vis,
        config.lsm,
        beam_type=config.beam_type,
        eb_ms=config.eb_ms,
        eb_coeffs=config.eb_coeffs,
    )  # .persist()

    # Call the solver
    refant = 0
    logger.info(f"Setting calibration in {config.fchunk}-channel chunks")
    logger.info(" - Using solver jones_substitution")
    logger.info(" - Using niter=20 and tol=1e-4 for this initial run")
    initialtable = run_solver(
        vis=vis,
        modelvis=modelvis,
        solver="jones_substitution",
        niter=20,
        tol=1e-4,
        refant=refant,
    )

    # Load all of the solutions into a numpy array
    logger.info("Running graph and returning calibration solutions")
    initialtable.load()

    # Fit for any differential rotations (single call for all channels).
    # Return 1D array of rotation measure values.
    logger.info("Fitting differential rotations")
    rm_est = model_rotations(initialtable, refant=refant, plot_sample=True)
    # rm_est = np.zeros(20)

    # Re-predict model visibilities
    logger.info("Re-predicting model vis with RM estimates")
    modelvis = predict_vis(
        vis,
        config.lsm,
        beam_type=config.beam_type,
        eb_ms=config.eb_ms,
        eb_coeffs=config.eb_coeffs,
        station_rm=rm_est,
    )  # .persist()

    logger.info(f"Resetting calibration in {config.fchunk}-channel chunks")
    logger.info(" - First using solver jones_substitution again")
    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        solver="jones_substitution",
        niter=20,
        tol=1e-4,
        refant=refant,
    )
    logger.info(" - Then improving using solver normal_equations")
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

    # Plot a sample of the results
    _, axs = plt.subplots(3, 3, figsize=(14, 14), sharey=True)
    # plot stations at a low RM, the median RM and the max RM
    x = gaintable.frequency.data / 1e6
    stns = np.abs(rm_est).argsort()[[len(rm_est) // 4, len(rm_est) // 2, -1]]
    for k, stn in enumerate(stns):
        J = initialtable.gain.data[0, stn] @ np.linalg.inv(
            initialtable.gain.data[0, refant, ..., :, :]
        )
        ax = axs[k, 0]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            ax.plot(x, np.imag(J[:, p // 2, p % 2]), f"C{pol}--")
        ax.set_title(f"Bandpass for station {stn} (rel to 0) (re: -, im: --)")
        ax.grid()
        ax.legend()

        J = generate_rotation_matrices(rm_est, gaintable.frequency.data)[stn]
        ax = axs[k, 1]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            ax.plot(x, np.imag(J[:, p // 2, p % 2]), f"C{pol}--")
        ax.set_title(f"Bandpass RM model, RM = {rm_est[stn]:.3f}")
        ax.grid()
        ax.legend()

        J = gaintable.gain.data[0, stn] @ np.linalg.inv(
            gaintable.gain.data[0, refant, ..., :, :]
        )
        ax = axs[k, 2]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            ax.plot(x, np.imag(J[:, p // 2, p % 2]), f"C{pol}--")
        ax.set_title("De-rotated bandpass")
        ax.grid()
        ax.legend()

    plt.savefig("bandpass_stages.png")

    if config.h5parm_name is not None:
        logger.info(f"Writing solutions to {config.h5parm_name}")
        export_gaintable_to_h5parm(gaintable, config.h5parm_name)
    if config.hdf5_name is not None:
        logger.info(f"Writing solutions to {config.hdf5_name}")
        export_gaintable_to_hdf5(gaintable, config.hdf5_name)

    # Convergence checks (noise-free demo version)
    #  - Note that this runs the graph again.
    #  - This is just a quick check, so it shouldn't hurt to run it again.
    #  - RM matrices are now applied before the beam matrices.
    if config.do_simulation:
        logger.info("Checking results")
        vis = vis.load()
        modelvis.load()
        vis = apply_gaintable(vis=vis, gt=gaintable, inverse=True)
        diff = (vis.vis - modelvis.vis).data
        logger.info(f"model max = {np.max(np.abs(modelvis.vis.data)):.1f}")
        logger.info(f"corrected max = {np.max(np.abs(vis.vis.data)):.1f}")
        logger.info(f"diff max = {np.max(np.abs(diff)):.1e}")
        logger.info(
            "diff max (relative) = "
            + f"{np.max(np.abs(diff)) / np.max(np.abs(modelvis.vis.data)):.1e}"
        )
        if np.max(np.abs(diff)) / np.max(np.abs(modelvis.vis.data)) < 2e-4:
            logger.info("Convergence checks passed")
        else:
            logger.warning("Solving failed to converge")

    # Shut down the scheduler and workers
    client.close()
    if config.dask_scheduler_address is None:
        client.shutdown()
