"""Helper functions"""

import warnings

import numpy as np
import xarray as xr
from astropy import constants as const
from astropy.coordinates import Angle, SkyCoord

# from ska_sdp_func_python.calibration.operations import apply_gaintable
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_create import create_visibility
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.processing_tasks.lsm_tmp import (
    convert_model_to_skycomponents,
    generate_lsm,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    predict_from_components,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


logger = setup_logger("workflow.utils")


def create_demo_ms(
    ms_name: str = "demo.ms",
    gains: bool = True,
    leakage: bool = False,
    rotation: bool = False,
    wide_channels: bool = False,
    nchannels: int = 64,
    ntimes: int = 1,
    fov: float = 10,
    flux_limit: float = 1,
    gleamfile: str = None,
    eb_ms: str = None,
    eb_coeffs: str = None,
) -> xr.Dataset:
    """Create a demo Visibility dataset and write to a MSv2 file.

    Using the ECP-240228 modified AA2 array.

    Should have an option to add sample noise.

    :param ms_name: Name of output Measurement Set.
    :param gains: Whether to include DI antenna gain terms (def=True).
    :param leakage: Whether to include DI antenna leakage terms (def=False).
    :param rotation: Whether to include differential rotation (def=False).
    :param wide_channels: Use 781.25 kHz channels? Default is False (5.4 kHz).
    :param nchannels: Number of channels. Default is 64.
    :param ntimes: Number of time steps. Default is 1.
    :param fov: Field of view width to use when generating the sky model
    :param flux_limit: flux limit to use when generating the sky model
    :param gleamfile: Pathname of GLEAM catalogue gleamegc.dat.
    :param eb_ms: Pathname of Everybeam mock Measurement Set.
    :param eb_coeffs: Path to Everybeam coeffs directory.
    :return: GainTable applied to data
    """
    # Check input
    if gleamfile is None:
        raise ValueError("GLEAM catalogue gleamegc.dat is required.")
    if eb_ms is None:
        raise ValueError("Pathname of Everybeam mock MS is required.")
    if eb_coeffs is None:
        raise ValueError("Path to Everybeam coeffs directory is required.")

    # Set up the array
    #  - Read in an array configuration
    low_config = create_named_configuration("LOWBD2")

    #  - Down-select to a desired sub-array
    #     - ECP-240228 modified AA2 clusters:
    #         Southern Arm: S8 (x6), S9, S10 (x6), S13, S15, S16
    #         Northern Arm: N8, N9, N10, N13, N15, N16
    #         Eastern Arm: E8, E9, E10, E13.
    #     - Most include only 4 of 6 stations, so just use the first 4:
    AA2 = (
        np.concatenate(
            (
                345 + np.arange(6),  # S8-1:6
                351 + np.arange(4),  # S9-1:4
                429 + np.arange(6),  # S10-1:6
                447 + np.arange(4),  # S13-1:4
                459 + np.arange(4),  # S15-1:4
                465 + np.arange(4),  # S16-1:4
                375 + np.arange(4),  # N8-1:4
                381 + np.arange(4),  # N9-1:4
                471 + np.arange(4),  # N10-1:4
                489 + np.arange(4),  # N13-1:4
                501 + np.arange(4),  # N15-1:4
                507 + np.arange(4),  # N16-1:4
                315 + np.arange(4),  # E8-1:4
                321 + np.arange(4),  # E9-1:4
                387 + np.arange(4),  # E10-1:4
                405 + np.arange(4),  # E13-1:4
            )
        )
        - 1
    )
    mask = np.isin(low_config.id.data, AA2)
    nstations = low_config.stations.shape[0]
    low_config = low_config.sel(indexers={"id": np.arange(nstations)[mask]})

    #  - Reset relevant station parameters
    nstations = low_config.stations.shape[0]
    low_config.stations.data = np.arange(nstations).astype("str")
    low_config = low_config.assign_coords(id=np.arange(nstations))
    low_config.attrs["name"] = "AA2-Low-ECP-240228"

    logger.info(f"Using {low_config.name} with {nstations} stations")

    # Set up the observation
    #  - Set the phase centre in the ICRS coordinate frame
    ra0 = Angle(0.0, unit="degree")
    dec0 = Angle(-27.0, unit="degree")

    #  - Set the parameters of sky model components
    if wide_channels:
        chanwidth = 781.25e3  # Hz
    else:
        chanwidth = 5.4e3  # Hz
    nfrequency = nchannels
    frequency = 781.25e3 * 128 + chanwidth * np.arange(nfrequency)
    sample_time = 0.9  # seconds
    solution_interval = ntimes * sample_time

    #  - Set the phase centre hour angle range for the sim (in radians)
    ha0 = 1 * np.pi / 12  # radians
    ha = ha0 + np.arange(0, solution_interval, sample_time) / 3600 * np.pi / 12

    # Create an zeroed Visibility dataset
    vis = create_visibility(
        low_config,
        ha,
        frequency,
        channel_bandwidth=[chanwidth] * len(frequency),
        polarisation_frame=PolarisationFrame("linear"),
        phasecentre=SkyCoord(ra=ra0, dec=dec0),
        weight=1.0,
    )

    # Generate the Local Sky Model
    lsm = generate_lsm(
        gleamfile=gleamfile,
        phasecentre=vis.phasecentre,
        fov=fov,
        flux_limit=flux_limit,
    )

    logger.info(f"Using {len(lsm)} components from {gleamfile}")

    # Convert the LSM to a Skycomponents list
    lsm_components = convert_model_to_skycomponents(lsm, vis.frequency.data)

    # Predict visibilities and add to the vis dataset
    predict_from_components(
        vis, lsm_components, eb_coeffs=eb_coeffs, eb_ms=eb_ms
    )

    # Generate direction-independent random complex antenna Jones matrices
    jones = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )
    if gains:
        logger.info("Applying direction-independent gain corruptions")
        g_sigma = 0.1
        jones.gain.data[..., 0, 0] = (
            np.random.normal(1, g_sigma, jones.gain.shape[:3])
            + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
        )
        jones.gain.data[..., 1, 1] = (
            np.random.normal(1, g_sigma, jones.gain.shape[:3])
            + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
        )
    if leakage:
        # Should perhaps do the proper multiplication with the gains.
        # Will do if other effects like xy-phase are added.
        logger.info("Applying direction-independent leakage corruptions")
        g_sigma = 0.1
        jones.gain.data[..., 0, 1] = (
            np.random.normal(0, g_sigma, jones.gain.shape[:3])
            + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
        )
        jones.gain.data[..., 1, 0] = (
            np.random.normal(0, g_sigma, jones.gain.shape[:3])
            + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
        )
    if rotation:
        logger.info("Applying DI lambda^2-dependent rotations")
        # Not particularly realistic Faraday rotation gradient across the array
        x = low_config.xyz.data[:, 0]
        pp_rm = 1 + 4 * (x - np.min(x)) / (np.max(x) - np.min(x))
        lambda_sq = (
            const.c.value / frequency  # pylint: disable=no-member
        ) ** 2
        for stn in range(nstations):
            d_pa = pp_rm[stn] * lambda_sq
            fr_mat = np.stack(
                (np.cos(d_pa), -np.sin(d_pa), np.sin(d_pa), np.cos(d_pa)),
                axis=1,
            ).reshape(-1, 2, 2)
            tmp = jones.gain.data[:, stn].copy()
            jones.gain.data[:, stn] = np.einsum("tfpx,fxq->tfpq", tmp, fr_mat)

    # Apply Jones matrices to the dataset
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)

    # Export vis to the file
    export_visibility_to_ms(ms_name, [vis])

    return jones
