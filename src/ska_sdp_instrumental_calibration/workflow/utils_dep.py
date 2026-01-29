from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility.vis_create import create_visibility
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    Component,
    LocalSkyComponent,
)
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    predict_from_components,
)

logger = setup_logger(__name__)


def convert_model_to_skycomponents(
    model: list[Component], freq
) -> list[LocalSkyComponent]:
    """Convert the LocalSkyModel to a list of SkyComponents.

    All sources are unpolarised and specified in the linear polarisation frame
    using XX = YY = Stokes I.

    Function :func:`~deconvolve_gaussian` is used to deconvolve the MWA
    synthesised beam from catalogue shape parameters of each component.
    Components with non-zero widths after this process are stored with
    shape = "GAUSSIAN". Otherwise shape = "POINT".

    :param model: Component list
    :param freq: Frequency list in Hz
    :param freq0: Reference Frequency for flux scaling in Hz. Default is 200e6.
        Note: freq0 should really be part of the sky model
    :return: SkyComponent list
    """

    freq = np.array(freq)
    return [
        LocalSkyComponent.create_from_component(comp, freq) for comp in model
    ]


def create_bandpass_table(vis: xr.Dataset) -> xr.Dataset:
    """Create full-length but unset gaintable for bandpass solutions.

    :param vis: Visibility dataset containing metadata.
    :return: GainTable dataset
    """
    jones_type = "B"

    soln_int = np.max(vis.time.data) - np.min(vis.time.data)
    # Function solve_gaintable can ignore the last sample if gain_interval is
    # exactly soln_int. So make it a little bigger.
    gain_interval = [soln_int * 1.00001]
    gain_time = np.array([np.average(vis.time)])
    ntimes = len(gain_time)
    if ntimes != 1:
        raise ValueError("expect single time step in bandpass table")

    nants = vis.visibility_acc.nants

    gain_frequency = vis.frequency.data
    nfrequency = len(gain_frequency)

    receptor_frame = ReceptorFrame(vis.visibility_acc.polarisation_frame.type)
    nrec = receptor_frame.nrec

    gain_shape = [ntimes, nants, nfrequency, nrec, nrec]
    gain = da.ones(gain_shape, dtype=np.complex64)
    if nrec > 1:
        gain[..., 0, 1] = da.zeros(gain_shape[:3], dtype=np.complex64)
        gain[..., 1, 0] = da.zeros(gain_shape[:3], dtype=np.complex64)

    gain_weight = da.ones(gain_shape, dtype=np.float32)
    gain_residual = da.zeros(
        [ntimes, nfrequency, nrec, nrec], dtype=np.float32
    )

    gain_table = GainTable.constructor(
        gain=gain,
        time=gain_time,
        interval=gain_interval,
        weight=gain_weight,
        residual=gain_residual,
        frequency=gain_frequency,
        receptor_frame=receptor_frame,
        phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        jones_type=jones_type,
    )

    return gain_table


def create_demo_ms(
    ms_name: str = "demo.ms",
    delays: bool = False,
    gains: bool = True,
    leakage: bool = False,
    rotation: bool = False,
    wide_channels: bool = False,
    nchannels: int = 64,
    ntimes: int = 1,
    phasecentre: SkyCoord = None,
    lsm: list[Component] = [],
    beam_type: str = "everybeam",
    eb_coeffs: Optional[str] = None,
    eb_ms: Optional[str] = None,
) -> xr.Dataset:
    """Create a demo Visibility dataset and write to a MSv2 file.

    Using the ECP-240228 modified AA2 array.

    Should have an option to add sample noise.

    :param ms_name: Name of output Measurement Set.
    :param delays: Whether to include DI antenna delay terms (def=False).
    :param gains: Whether to include DI antenna gain terms (def=True).
    :param leakage: Whether to include DI antenna leakage terms (def=False).
    :param rotation: Whether to include differential rotation (def=False).
    :param wide_channels: Use 781.25 kHz channels? Default is False (5.4 kHz).
    :param nchannels: Number of channels. Default is 64.
    :param ntimes: Number of time steps. Default is 1.
    :param lsm: Local sky model
    :param beam_type: Type of beam model to use. Default is "everybeam". If set
        to None, no beam will be applied.
    :param eb_coeffs: Everybeam coeffs datadir containing beam coefficients.
        Required if beam_type is "everybeam".
    :param eb_ms: Measurement set need to initialise the everybeam telescope.
        Required if bbeam_type is "everybeam".
    :return: GainTable applied to data
    """
    if phasecentre is None:
        raise ValueError("Parameter phasecentre is required")

    # Set up the array
    #  - Read in an array configuration
    low_config = create_named_configuration("LOWBD2")

    #  - Down-select to a desired sub-array
    #     - ECP-240228 modified AA2 clusters:
    #         Southern Arm: S8 (x6), S9, S10 (x6), S13, S15, S16
    #         Northern Arm: N8, N9, N10, N13, N15, N16
    #         Eastern Arm: E8, E9, E10, E13.
    #     - Most include only 4 of 6 stations, so just use the first 4:
    # AA2 = (
    #     np.concatenate(
    #         (
    #             345 + np.arange(6),  # S8-1:6
    #             351 + np.arange(4),  # S9-1:4
    #             429 + np.arange(6),  # S10-1:6
    #             447 + np.arange(4),  # S13-1:4
    #             459 + np.arange(4),  # S15-1:4
    #             465 + np.arange(4),  # S16-1:4
    #             375 + np.arange(4),  # N8-1:4
    #             381 + np.arange(4),  # N9-1:4
    #             471 + np.arange(4),  # N10-1:4
    #             489 + np.arange(4),  # N13-1:4
    #             501 + np.arange(4),  # N15-1:4
    #             507 + np.arange(4),  # N16-1:4
    #             315 + np.arange(4),  # E8-1:4
    #             321 + np.arange(4),  # E9-1:4
    #             387 + np.arange(4),  # E10-1:4
    #             405 + np.arange(4),  # E13-1:4
    #         )
    #     )
    #     - 1
    # )
    # mask = np.isin(low_config.id.data, AA2)

    # Change to AA1. I only know that it will include stations from clusters
    # S8, S9 and S10. Include S16 as well, but this will be updated once the
    # PI25 simulation config is confirmed.
    AA1 = (
        np.concatenate(
            (
                345 + np.arange(6),  # S8-1:6
                351 + np.arange(4),  # S9-1:4
                429 + np.arange(6),  # S10-1:6
                465 + np.arange(4),  # S16-1:4
            )
        )
        - 1
    )
    mask = np.isin(low_config.id.data, AA1)

    nstations = low_config.stations.shape[0]
    low_config = low_config.sel(indexers={"id": np.arange(nstations)[mask]})

    #  - Reset relevant station parameters
    nstations = low_config.stations.shape[0]
    low_config.stations.data = np.arange(nstations).astype("str")
    low_config = low_config.assign_coords(id=np.arange(nstations))
    # low_config.attrs["name"] = "AA2-Low-ECP-240228"
    low_config.attrs["name"] = "AA1-Low"

    logger.info(f"Using {low_config.name} with {nstations} stations")

    # Set up the observation

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
        phasecentre=phasecentre,
        weight=1.0,
    )

    # Convert the LSM to a Skycomponents list
    lsm_components = convert_model_to_skycomponents(lsm, vis.frequency.data)

    if rotation:
        logger.info("Applying DI lambda^2-dependent rotations during predict")
        # Not particularly realistic Faraday rotation
        # Just a DI term with some nominal variation across the array
        x = low_config.xyz.data[:, 0]
        rm = 0.5 * (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        rm = None

    # Predict visibilities and add to the vis dataset
    predict_from_components(
        vis=vis,
        skycomponents=lsm_components,
        beam_type=beam_type,
        eb_coeffs=eb_coeffs,
        eb_ms=eb_ms,
        station_rm=rm,
    )

    # Generate direction-independent random complex antenna Jones matrices
    jones = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )
    if gains:
        logger.info("Applying direction-independent gain corruptions")
        g_sigma = 0.05
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
        g_sigma = 0.01
        jones.gain.data[..., 0, 1] = (
            np.random.normal(0, g_sigma, jones.gain.shape[:3])
            + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
        )
        jones.gain.data[..., 1, 0] = (
            np.random.normal(0, g_sigma, jones.gain.shape[:3])
            + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
        )

    if delays:
        # Non-dispersive delays
        # Multiply the existing Jones matrices on the LHS
        stndelayX = 5e-9 * np.random.rand(nstations)
        stndelayY = stndelayX
        dl_mat = np.zeros((nstations, nfrequency, 2, 2), "complex")
        dl_mat[:, :, 0, 0] = np.exp(
            -2j * np.pi * np.einsum("s,f->sf", stndelayX, frequency)
        )
        dl_mat[:, :, 1, 1] = np.exp(
            -2j * np.pi * np.einsum("s,f->sf", stndelayY, frequency)
        )
        tmp = jones.gain.data.copy()
        jones.gain.data = np.einsum("sfpx,tsfxq->tsfpq", dl_mat, tmp)

    # Apply Jones matrices to the dataset
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)

    # Export vis to the file
    export_visibility_to_ms(ms_name, [vis])

    return jones
