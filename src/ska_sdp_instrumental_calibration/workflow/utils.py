"""Helper functions"""

import warnings
from typing import Iterable, Literal, Optional

import h5py
import numpy as np
import xarray as xr
from astropy import constants as const
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray

# from ska_sdp_func_python.calibration.operations import apply_gaintable
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.calibration.calibration_model import GainTable
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


def _ndarray_of_null_terminated_bytes(strings: Iterable[str]) -> NDArray:
    # NOTE: making antenna names one character longer, in keeping with
    # ska-sdp-batch-preprocess
    return np.asarray([s.encode("ascii") + b"\0" for s in strings])


def create_soltab_group(
    solset: h5py.Group, solution_type: Literal["amplitude", "phase"]
) -> h5py.Group:
    """Create soltab group under given solset group.

    :param solset: base-level HDF5 group to update
    :param solution_type: only "amplitude" and "phase" are supported at present
    :return: HDF5 group for the "solution_type" data
    """
    soltab = solset.create_group(f"{solution_type}000")
    soltab.attrs["TITLE"] = np.bytes_(solution_type)
    return soltab


def create_soltab_datasets(soltab: h5py.Group, gaintable: GainTable):
    """Add a dataset for each of the GainTable dimensions.

    :param soltab: HDF5 table to update
    :param gaintable: GainTable
    """
    # create a dataset for each dimension
    for dim in list(gaintable.gain.sizes):
        soltab.create_dataset(dim, data=gaintable[dim].data)

    # create datasets for the data and weights
    shape = gaintable.gain.shape
    axes = np.bytes_(",".join(list(gaintable.gain.sizes)))

    val = soltab.create_dataset("val", shape=shape, dtype=float)
    val.attrs["AXES"] = axes

    weight = soltab.create_dataset("weight", shape=shape, dtype=float)
    weight.attrs["AXES"] = axes

    return val, weight


def export_gaintable_to_h5parm(
    gaintable: GainTable, filename: str, squeeze: bool = False
):
    """Export a GainTable to a H5Parm HDF5 file.

    :param gaintable: GainTable
    :param filename: Name of H5Parm file
    :param squeeze: If True, remove axes of length one from dataset
    """
    logger.info(f"exporting cal solutions to {filename}")

    # check gaintable gain and weight dimensions
    dims = ["time", "antenna", "frequency", "receptor1", "receptor2"]
    if list(gaintable.gain.sizes) != dims:
        raise ValueError(f"Unexpected dims: {list(gaintable.gain.sizes)}")

    # adjust dimensions to be consistent with H5Parm output format
    gaintable = gaintable.rename({"antenna": "ant", "frequency": "freq"})
    gaintable = gaintable.stack(pol=("receptor1", "receptor2"))
    polstrs = _ndarray_of_null_terminated_bytes(
        [f"{p1}{p2}" for p1, p2 in gaintable["pol"].data]
    )
    gaintable = gaintable.assign_coords({"pol": polstrs})

    # check polarisations and discard unused products
    polstrs = _ndarray_of_null_terminated_bytes(["XX", "XY", "YX", "YY"])
    if not np.array_equal(gaintable["pol"].data, polstrs):
        raise ValueError("Subsequent pipelines assume linear pol order")
    if np.sum(np.abs(gaintable.isel(pol=[1, 2]).weight.data)) == 0:
        gaintable = gaintable.isel(pol=[0, 3])

    # replace antenna indices with antenna names
    if gaintable.configuration is None:
        raise ValueError("Missing gt config. H5Parm requires antenna names")
    antenna_names = _ndarray_of_null_terminated_bytes(
        gaintable.configuration.names.data[gaintable["ant"].data]
    )
    gaintable = gaintable.assign_coords({"ant": antenna_names})

    # remove axes of length one if required
    if squeeze:
        gaintable = gaintable.squeeze(drop=True)

    logger.info(f"output dimensions: {dict(gaintable.gain.sizes)}")

    with h5py.File(filename, "w") as file:

        solset = file.create_group("sol000")

        # Amplitude table
        soltab = create_soltab_group(solset, "amplitude")
        val, weight = create_soltab_datasets(soltab, gaintable)
        val[...] = np.absolute(gaintable["gain"].data)
        weight[...] = gaintable["weight"].data

        # Phase table
        soltab = create_soltab_group(solset, "phase")
        val, weight = create_soltab_datasets(soltab, gaintable)
        val[...] = np.angle(gaintable["gain"].data)
        weight[...] = gaintable["weight"].data
