"""Helper functions"""

__all__ = [
    "create_demo_ms",
    "create_bandpass_table",
    "get_ms_metadata",
    "get_phasecentre",
]

import warnings
from collections import namedtuple
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from astropy import constants as const
from astropy.coordinates import SkyCoord
from casacore.tables import table

# from ska_sdp_func_python.calibration.operations import apply_gaintable
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
from ska_sdp_datamodels.visibility.vis_io_ms import (
    create_visibility_from_ms,
    export_visibility_to_ms,
)

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.processing_tasks.lsm import (
    Component,
    convert_model_to_skycomponents,
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

    # Predict visibilities and add to the vis dataset
    predict_from_components(
        vis=vis,
        skycomponents=lsm_components,
        beam_type=beam_type,
        eb_coeffs=eb_coeffs,
        eb_ms=eb_ms,
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


def get_phasecentre(ms_name: str) -> SkyCoord:
    """Return the phase centre of a MSv2 Measurement Set.

    The first field is used if there more than one.

    :param ms_name: Name of input Measurement Set.
    :return: phase centre
    """
    fieldtab = table(f"{ms_name}/FIELD", ack=False)
    field = 0
    pc = fieldtab.getcol("PHASE_DIR")[field, 0, :]
    return SkyCoord(
        ra=pc[0], dec=pc[1], unit="radian", frame="icrs", equinox="J2000"
    )


def get_ms_metadata(ms_name: str) -> xr.Dataset:
    """Get Visibility dataset metadata.

    Fixme: use ska_sdp_datamodels.visibility.vis_io_ms.get_ms_metadata once
    YAN-1990 is finalised. For now, read a single channel and use its metadata.

    :param ms_name: Name of input Measurement Set
    :return: Namedtuple of metadata products required by Visibility.constructor
        - uvw
        - baselines
        - time
        - frequency
        - channel_bandwidth
        - integration_time
        - configuration
        - phasecentre
        - polarisation_frame
        - source
        - meta
    """
    # Read a single-channel from the dataset
    tmpvis = create_visibility_from_ms(ms_name, start_chan=0, end_chan=0)[0]
    # Update frequency metadata for the full dataset
    spwtab = table(f"{ms_name}/SPECTRAL_WINDOW", ack=False)
    frequency = np.array(spwtab.getcol("CHAN_FREQ")[0])
    channel_bandwidth = np.array(spwtab.getcol("CHAN_WIDTH")[0])

    ms_metadata = namedtuple(
        "ms_metadata",
        [
            "uvw",
            "baselines",
            "time",
            "frequency",
            "channel_bandwidth",
            "integration_time",
            "configuration",
            "phasecentre",
            "polarisation_frame",
            "source",
            "meta",
        ],
    )

    return ms_metadata(
        uvw=tmpvis.uvw.data,
        baselines=tmpvis.baselines,
        time=tmpvis.time,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        integration_time=tmpvis.integration_time,
        configuration=tmpvis.configuration,
        phasecentre=tmpvis.phasecentre,
        polarisation_frame=PolarisationFrame(tmpvis._polarisation_frame),
        source="bpcal",
        meta=None,
    )


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
