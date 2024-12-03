"""Wrapper functions to manage xarray dataset map_blocks calls.

Topics from Vincent's MR review to be considered in ongoing work:
 - [extract map_blocks functions] VM: "I don't recommend calling map_blocks and
   other dask constructs on functions defined at such local scope, because
   there's always a danger they will capture some of their context in a
   closure, and that context (which might include large arrays) then has to be
   passed around between the workers and the scheduler as part of tasks." I
   agree with this change.
 - [pure functions preferred] VM: "mutating the input Visibility dataset is not
   necessary and is more trouble than worth. This would all be cleaner if you
   returned a new visibility, and xarray + dask are smart enough to optimally
   reuse all data variables common to both the input and (new) output
   visibility." This needs more discussion. In a number of cases I do want to
   update the input dataset and I would prefer not to make a copy of the data.
   For instance when accumulating large model visibility datasets. Also, most
   of the functions lie between non-pure map_blocks calls and non-pure
   sdp-func-python calls, so it seems reasonable to be consistent.
 - [new data models] VM: "it could be dangerous to refer to dimensions by index
   on xarray Datasets, unless the dimension order (xds.dims) is known with
   certainty (note that dimension order in xarray is different from data order
   in memory, both can change independently, because it's all numpy strides
   under the hood anyway). The safe way is to refer to dimensions by name..."
"""

__all__ = [
    "load_ms",
    "predict_vis",
    "run_solver",
    "apply_gaintable_to_dataset",
    "simplify_baselines_dim",
    "restore_baselines_dim",
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
from ska_sdp_datamodels.visibility import Visibility
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
    vis = simplify_baselines_dim(
        Visibility.constructor(
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
        )
    ).chunk({"frequency": fchunk})

    # Set up function for map_blocks
    def _load(vischunk, ms_name, frequency):
        if len(vischunk.frequency) > 0:
            start = np.where(frequency == vischunk.frequency.data[0])[0][0]
            end = np.where(frequency == vischunk.frequency.data[-1])[0][0]
            return simplify_baselines_dim(
                create_visibility_from_ms(
                    ms_name,
                    start_chan=start,
                    end_chan=end,
                )[0]
            )

    # Call map_blocks function and return result
    return vis.map_blocks(_load, args=[ms_name, frequency], template=vis)


def predict_vis(
    vis: xr.Dataset,
    lsm: list,
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
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
    def _predict(vischunk, lsm, beam_type, eb_coeffs, eb_ms):
        if len(vischunk.frequency) > 0:
            # Evaluate LSM for current band
            lsm_components = convert_model_to_skycomponents(
                lsm, vischunk.frequency.data, freq0=200e6
            )
            # Switch to standard variable names and coords for the SDP call
            vischunk = restore_baselines_dim(vischunk)
            # Call predict
            predict_from_components(
                vischunk,
                lsm_components,
                beam_type=beam_type,
                eb_coeffs=eb_coeffs,
                eb_ms=eb_ms,
            )
            # Change variable names back for map_blocks I/O checks
            vischunk = simplify_baselines_dim(vischunk)
        return vischunk

    # Call map_blocks function and return result
    return modelvis.map_blocks(
        _predict, args=[lsm, beam_type, eb_coeffs, eb_ms]
    )


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
            # Switch to standard variable names and coords for the SDP call
            gainchunk = gainchunk.rename({"soln_time": "time"})
            vischunk = restore_baselines_dim(vischunk)
            modelchunk = restore_baselines_dim(modelchunk)
            # Call solver
            solve_bandpass(
                vis=vischunk,
                modelvis=modelchunk,
                gain_table=gainchunk,
                solver=solver,
                refant=refant,
                niter=niter,
            )
            # Change the time dimension name back for map_blocks I/O checks
            gainchunk = gainchunk.rename({"time": "soln_time"})

        return gainchunk

    # map_blocks won't accept dimensions that differ but have the same name
    # So rename the gain time dimension (and coordinate)
    gaintable = gaintable.rename({"time": "soln_time"})
    gaintable = gaintable.map_blocks(_solve, args=[vis, modelvis, refant])
    # Undo any temporary variable name changes
    gaintable = gaintable.rename({"soln_time": "time"})
    return gaintable


def apply_gaintable_to_dataset(
    vis: xr.Dataset,
    gaintable: xr.Dataset,
    inverse: bool = False,
) -> xr.Dataset:
    """Do the bandpass calibration.

    :param vis: Chunked Visibility dataset containing observed data.
    :param gaintable: Chunked Visibility dataset containing model data.
    :param inverse: Apply the inverse. This requires the gain matrices to be
        square. (default=False)
    :return: Calibrated Visibility dataset
    """

    # Set up function for map_blocks
    def _apply(vischunk, gainchunk, inverse):
        if len(vischunk.frequency) > 0:
            if np.any(gainchunk.frequency.data != vischunk.frequency.data):
                raise ValueError("Inconsistent frequencies")
            # Switch back to standard variable names for the SDP call
            gainchunk = gainchunk.rename({"soln_time": "time"})
            # Call apply function
            vischunk = apply_gaintable(
                vis=vischunk, gt=gainchunk, inverse=inverse
            )
        return vischunk

    # map_blocks won't accept dimensions that differ but have the same name
    # So rename the gain time dimension (and coordinate)
    gaintable = gaintable.rename({"time": "soln_time"})
    return vis.map_blocks(_apply, args=[gaintable, inverse])


def simplify_baselines_dim(vis: xr.Dataset) -> xr.Dataset:
    """Move the baselines coord to a data variable and use an index instead.

    A number of xarray/dask operations exit in error, complaining about the
    pandas.MultiIndex baselines coordinate of Visibility datasets. In most
    cases this can be avoided by replacing the baselines coordinate with a
    more simple coordinate and resetting baselines as a data variable.

    :param vis: Standard Visibility dataset
    :return: Modified Visibility dataset
    """
    if vis.coords.get("baselines") is None:
        logger.warning("No baselines coord in dataset. Returning unchanged")
        return vis
    else:
        logger.debug("Swapping baselines MultiIndex coord with indices")
        if vis.variables.get("baselineid") is None:
            vis = vis.assign_coords(
                baselineid=("baselines", np.arange(len(vis.baselines)))
            )
        return vis.swap_dims({"baselines": "baselineid"}).reset_coords(
            ("baselines", "antenna1", "antenna2")
        )


def restore_baselines_dim(vis: xr.Dataset) -> xr.Dataset:
    """Move the baselines data variable back to a coordinate.

    Reverse of simplify_baselines_dim, needed for some SDP functions.

    :param vis: Modified Visibility dataset
    :return: Standard Visibility dataset
    """
    if vis.coords.get("baselineid") is None:
        logger.warning("No baselineid coord in dataset. Returning unchanged")
        return vis
    elif vis.coords.get("baselines") is not None:
        logger.warning("Coord baselines already exists. Returning unchanged")
        return vis
    else:
        logger.debug("Restoring baselines MultiIndex coord")
        return vis.swap_dims({"baselineid": "baselines"}).reset_coords(
            "baselineid", drop=True
        )