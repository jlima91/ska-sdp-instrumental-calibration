"""Wrapper functions to manage xarray dataset map_blocks calls.

General comment from Vincent: I don't recommend calling map_blocks and other
dask constructs on functions defined at such local scope, because there's
always a danger they will capture some of their context in a closure, and that
context (which might include large arrays) then has to be passed around between
the workers and the scheduler as part of tasks. Here you're not using variables
from a higher scope in that local function, so you should be fine, but a
mistake happens easily.
"""

__all__ = [
    "load_ms",
    "predict_vis",
    "run_solver",
    "apply_gaintable_to_dataset",
]

from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from casacore.tables import table
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame

# from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
    solve_bandpass,
)
from ska_sdp_instrumental_calibration.processing_tasks.lsm_tmp import (
    convert_model_to_skycomponents,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    predict_from_components,
)
from ska_sdp_instrumental_calibration.vis_model import Visibility

logger = setup_logger("data_managers.dask_wrappers")


def load_ms(ms_name: str, fchunk: int) -> xr.Dataset:
    """Distributed load of a MSv2 Measurement Set into a Visibility dataset.

    :param ms_name: Name of input Measurement Set.
    :param fchunk: Number of channels in the frequency chunks
    :return: Chunked Visibility dataset
    """
    # Get observation metadata
    #  - Frequency metadata
    spwtab = table(f"{ms_name}/SPECTRAL_WINDOW", ack=False)
    frequency = np.array(spwtab.getcol("CHAN_FREQ")[0])
    channel_bandwidth = np.array(spwtab.getcol("CHAN_WIDTH")[0])
    #  - Fixme: Change this to extract other metadata as done for frequency
    #  - For now read a single channel and use for all but frequency
    tmpvis = create_visibility_from_ms(ms_name, start_chan=0, end_chan=1)[0]
    shape = list(tmpvis.vis.shape)
    shape[2] = len(frequency)
    # Create a chunked dataset.
    vis = Visibility.constructor(
        configuration=tmpvis.configuration,
        phasecentre=tmpvis.phasecentre,
        time=tmpvis.time,
        integration_time=tmpvis.integration_time,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        polarisation_frame=PolarisationFrame(tmpvis._polarisation_frame),
        source="bpcal",
        meta=None,
        vis=da.zeros(shape, "complex"),
        weight=da.zeros(shape, "float"),
        flags=da.zeros(shape, "bool"),
        uvw=tmpvis.uvw.data,
        baselines=tmpvis.baselines,
    ).chunk({"frequency": fchunk})

    # Set up function for map_blocks
    def _load(vischunk, ms_name, frequency):
        if len(vischunk.frequency) > 0:
            start = np.where(frequency == vischunk.frequency.data[0])[0][0]
            end = np.where(frequency == vischunk.frequency.data[-1])[0][0]
            tmpvis = create_visibility_from_ms(
                ms_name,
                start_chan=start,
                end_chan=end,
            )[0]
            vischunk = Visibility.constructor(
                configuration=tmpvis.configuration,
                phasecentre=tmpvis.phasecentre,
                time=tmpvis.time,
                integration_time=tmpvis.integration_time,
                frequency=tmpvis.frequency.data,
                channel_bandwidth=tmpvis.channel_bandwidth.data,
                polarisation_frame=PolarisationFrame(
                    tmpvis._polarisation_frame
                ),
                source="bpcal",
                meta=None,
                vis=tmpvis.vis.data,
                weight=tmpvis.weight.data,
                flags=tmpvis.flags.data,
                uvw=tmpvis.uvw.data,
                baselines=tmpvis.baselines,
            )
        return vischunk

    # Call map_blocks function and return result
    return vis.map_blocks(_load, args=[ms_name, frequency], template=vis)


def predict_vis(
    vis: xr.Dataset,
    lsm: list,
    eb_ms: str,
    eb_coeffs: str,
) -> xr.Dataset:
    """Distributed load of a MSv2 Measurement Set into a Visibility dataset.

    :param vis: Visibility dataset containing observed data to be modelled.
    :param lsm: Number of channels in the frequency chunks.
    :param eb_ms: Pathname of Everybeam mock Measurement Set.
    :param eb_coeffs: Path to Everybeam coeffs directory.
    :return: Chunked Visibility dataset
    """
    # Create an empty model Visibility dataset
    modelvis = vis.assign({"vis": xr.zeros_like(vis.vis)})

    # Set up function for map_blocks
    def _predict(vischunk, lsm, eb_coeffs, eb_ms):
        if len(vischunk.frequency) > 0:
            # Evaluate LSM for current band
            lsm_components = convert_model_to_skycomponents(
                lsm, vischunk.frequency.data, freq0=200e6
            )
            # Call predict
            predict_from_components(
                vischunk, lsm_components, eb_coeffs=eb_coeffs, eb_ms=eb_ms
            )
        return vischunk

    # Call map_blocks function and return result
    return modelvis.map_blocks(_predict, args=[lsm, eb_coeffs, eb_ms])


def run_solver(
    vis: xr.Dataset,
    modelvis: xr.Dataset,
    gaintable: Optional[xr.Dataset] = None,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
) -> xr.Dataset:
    """Do the bandpass calibration.

    :param vis: Chunked Visibility dataset containing observed data.
    :param modelvis: Chunked Visibility dataset containing model data.
    :param gaintable: Optional chunked GainTable dataset containing initial
        solutions.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0).
    :param niter: Number of solver iterations (defaults to 50).

    :return: Chunked GainTable dataset
    """
    # Create a full-band bandpass calibration gain table
    #  - It may be more efficient to do this in sub-bands then concatenate...
    solution_interval = np.max(vis.time.data) - np.min(vis.time.data)
    if gaintable is None:
        fchunk = vis.chunks["frequency"][0]
        if fchunk <= 0:
            logger.warning("vis dataset does not appear to be chunked")
            fchunk = 1
        gaintable = create_gaintable_from_visibility(
            vis, jones_type="B", timeslice=solution_interval
        ).chunk({"frequency": fchunk})

    if len(gaintable.time) != 1:
        raise ValueError("error setting up gaintable")

    if refant is not None:
        if refant < 0 or refant >= len(gaintable.antenna):
            raise ValueError(f"invalid refant: {refant}")

    # Set up function for map_blocks
    def _solve(gainchunk, vischunk, modelchunk, refant):
        if len(vischunk.frequency) > 0:
            if np.any(gainchunk.frequency.data != vischunk.frequency.data):
                raise ValueError("Inconsistent frequencies")
            if np.any(gainchunk.frequency.data != modelchunk.frequency.data):
                raise ValueError("Inconsistent frequencies")
            solve_bandpass(
                vis=vischunk,
                modelvis=modelchunk,
                gain_table=gainchunk,
                solver=solver,
                refant=refant,
                niter=niter,
            )

        return gainchunk

    return gaintable.map_blocks(_solve, args=[vis, modelvis, refant])


def apply_gaintable_to_dataset(
    vis: xr.Dataset,
    gaintable: xr.Dataset,
) -> xr.Dataset:
    """Do the bandpass calibration.

    :param vis: Chunked Visibility dataset containing observed data.
    :param gaintable: Chunked Visibility dataset containing model data.
    :return: Calibrated Visibility dataset
    """

    # Set up function for map_blocks
    def _apply(vischunk, gainchunk):
        if len(vischunk.frequency) > 0:
            if np.any(gainchunk.frequency.data != vischunk.frequency.data):
                raise ValueError("Inconsistent frequencies")
            vischunk = apply_gaintable(
                vis=vischunk, gt=gainchunk, inverse=True
            )
        return vischunk

    return vis.map_blocks(_apply, args=[gaintable])
