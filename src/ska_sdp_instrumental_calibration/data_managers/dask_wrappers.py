"""Wrapper functions to manage xarray dataset map_blocks calls.

Topics from Vincent's MR review to be considered in ongoing work:
 - "mutating the input Visibility dataset is not necessary and is more trouble
   than worth. This would all be cleaner if you returned a new visibility, and
   xarray + dask are smart enough to optimally reuse all data variables common
   to both the input and (new) output visibility."
   This needs more discussion. In a number of cases I do want to update the
   input dataset and I would prefer not to make a copy of the data. For
   instance when accumulating large model visibility datasets.
 - "it could be dangerous to refer to dimensions by index on xarray Datasets,
   unless the dimension order (xds.dims) is known with certainty (note that
   dimension order in xarray is different from data order in memory, both can
   change independently, because it's all numpy strides under the hood anyway).
   The safe way is to refer to dimensions by name..."
"""

__all__ = [
    "load_ms",
    "predict_vis",
    "prediction_central_beams",
    "run_solver",
    "ingest_predict_and_solve",
    "apply_gaintable_to_dataset",
    "simplify_baselines_dim",
    "restore_baselines_dim",
]

import warnings

# avoid FutureWarnings that are logged from every DASK task.
# flake8: noqa: E402  # stop errors about this line coming before imports
warnings.simplefilter(action="ignore", category=FutureWarning)

from typing import Literal, Optional, Union

import dask.array as da
import numpy as np
import numpy.typing as npt
import xarray as xr
from casacore.tables import table

# avoid ska_sdp_datamodels/visibility/vis_model.py:201: FutureWarning: the
# `pandas.MultiIndex` object(s) passed as 'baselines' coordinate(s) or data
# variable(s) will no longer be implicitly promoted ...
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms

from ska_sdp_instrumental_calibration.data_managers.sky_model import Component
from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
    solve_bandpass,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    generate_central_beams,
    predict_from_components,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    convert_model_to_skycomponents,
    create_bandpass_table,
    get_ms_metadata,
)

logger = setup_logger("data_managers.dask_wrappers")


def _load(
    vischunk: xr.Dataset,
    ms_name: str,
    frequency: npt.NDArray[float],
) -> xr.Dataset:
    """Call create_visibility_from_ms.

    :param vis: Visibility dataset to be loaded.
    :param ms_name: Name of input Measurement Set.
    :param frequency: list of all frequencies in the MSv2 dataset.
    :return: Loaded Visibility dataset
    """
    if len(vischunk.frequency) > 0:
        start = np.where(frequency == vischunk.frequency.data[0])[0][0]
        end = np.where(frequency == vischunk.frequency.data[-1])[0][0]
        msvis = simplify_baselines_dim(
            create_visibility_from_ms(
                ms_name,
                start_chan=start,
                end_chan=end,
            )[0]
        )
        # Fixme: remove reassignment once YAN-1990 is finalised
        #   current version sets vis: complex128, weight: float64, flags: int64
        return msvis.assign(
            {
                "vis": msvis.vis.astype(np.complex64),
                "weight": msvis.weight.astype(np.float32),
                "flags": msvis.flags.astype(bool),
            }
        )


def load_ms(
    ms_name: str,
    fchunk: int,
    ack: bool = False,
    start_chan: int = 0,
    end_chan: int = 0,
    datacolumn: str = "DATA",
    selected_sources: list = None,
    selected_dds: list = None,
    average_channels: bool = False,
) -> xr.Dataset:
    """Distributed load of a MSv2 Measurement Set into a Visibility dataset.

    :param ms_name: Name of input Measurement Set.
    :param fchunk: Number of channels in the frequency chunks
    :param ack: Ask casacore to acknowledge each table operation
    :param start_chan: Starting channel to read
    :param end_chan: End channel to read4
    :param datacolumn: MS data column to read DATA, CORRECTED_DATA, MODEL_DATA
    :param selected_sources: Sources to select
    :param selected_dds: Data descriptors to select
    :param average_channels: Average all channels read
    :return: Chunked Visibility dataset
    """
    # Get observation metadata
    ms_metadata = get_ms_metadata(
        ms_name,
        start_chan=start_chan,
        ack=ack,
        datacolumn=datacolumn,
        end_chan=end_chan,
        selected_sources=selected_sources,
        selected_dds=selected_dds,
        average_channels=average_channels,
    )
    shape = (
        len(ms_metadata.time),
        len(ms_metadata.baselines),
        len(ms_metadata.frequency),
        len(PolarisationFrame.fits_codes[ms_metadata.polarisation_frame.type]),
    )

    # Create a chunked dataset.
    # Solving is done separately for each chunk, so make sure all times,
    # baselines and polarisations are available in each.
    vis = simplify_baselines_dim(
        Visibility.constructor(
            uvw=ms_metadata.uvw,
            baselines=ms_metadata.baselines,
            time=ms_metadata.time,
            frequency=ms_metadata.frequency,
            channel_bandwidth=ms_metadata.channel_bandwidth,
            vis=da.zeros(shape, dtype=np.complex64),
            weight=da.zeros(shape, dtype=np.float32),
            flags=da.zeros(shape, dtype=bool),
            integration_time=ms_metadata.integration_time,
            configuration=ms_metadata.configuration,
            phasecentre=ms_metadata.phasecentre,
            polarisation_frame=ms_metadata.polarisation_frame,
            source=ms_metadata.source,
            meta=ms_metadata.meta,
            # Fixme: use new default once YAN-1990 is finalised
            low_precision="float32",
        ).chunk(
            {
                "time": shape[0],
                "baselines": shape[1],
                "frequency": fchunk,
                "polarisation": shape[3],
            }
        )
    )
    # Fixme: remove reassignment once YAN-1990 is finalised
    #   current version sets vis: input, weight: low_precision, flags: int64
    vis = vis.assign({"flags": xr.zeros_like(vis.flags, dtype=bool)})

    # Call map_blocks function and return result
    return vis.map_blocks(
        _load, args=[ms_name, ms_metadata.frequency], template=vis
    )


def _predict(
    vischunk: xr.Dataset,
    lsm: list[Component],
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
    station_rm: Optional[xr.DataArray | npt.NDArray[np.float64]] = None,
    reset_vis: bool = False,
) -> xr.Dataset:
    """Call predict_from_components inside map_blocks.

    :param vis: Visibility dataset containing observed data to be modelled.
    :param lsm: List of LSM components. This is an intermediate format between
        the GSM and the evaluated SkyComponent list.
    :param beam_type: Type of beam model to use. Default is "everybeam".
    :param eb_ms: Pathname of Everybeam mock Measurement Set.
    :param eb_coeffs: Path to Everybeam coeffs directory.
    :param station_rm: Station rotation measure values. Default is None.
    :param reset_vis: Whether or not to set visibilities to zero before
        accumulating components. Default is False.
    :return: Predicted Visibility dataset
    """
    if len(vischunk.frequency) > 0:
        # Evaluate LSM for current band
        lsm_components = convert_model_to_skycomponents(
            lsm, vischunk.frequency.data
        )
        # Switch to standard variable names and coords for the SDP call
        vischunk = restore_baselines_dim(vischunk)

        # Dask array wrapped in xarray.Datarray
        if type(station_rm) == xr.DataArray:
            # should be numpy array after map_blocks
            station_rm = station_rm.data

        # Call predict
        predict_from_components(
            vischunk,
            lsm_components,
            beam_type=beam_type,
            eb_coeffs=eb_coeffs,
            eb_ms=eb_ms,
            station_rm=station_rm,
            reset_vis=reset_vis,
        )
        # Change variable names back for map_blocks I/O checks
        # Fixme: remove reassignment once YAN-1990 is finalised
        #   current version sets vis: complex128, weight: float64, flags: int64
        vischunk = simplify_baselines_dim(
            vischunk.assign(
                {
                    "vis": vischunk.vis.astype(np.complex64),
                    "weight": vischunk.weight.astype(np.float32),
                    "flags": vischunk.flags.astype(bool),
                }
            )
        )

    return vischunk


def predict_vis(
    vis: xr.Dataset,
    lsm: list,
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
    station_rm: Optional[npt.NDArray[np.float64] | da.Array] = None,
) -> xr.Dataset:
    """Distributed Visibility predict.

    :param vis: Visibility dataset containing observed data to be modelled.
        Should be chunked in frequency.
    :param lsm: List of LSM components. This is an intermediate format between
        the GSM and the evaluated SkyComponent list.
    :param beam_type: Type of beam model to use. Default is "everybeam".
    :param eb_ms: Pathname of Everybeam mock Measurement Set.
    :param eb_coeffs: Path to Everybeam coeffs directory.
    :param station_rm: Station rotation measure values. Default is None.
    :return: Predicted Visibility dataset
    """
    # Create an empty model Visibility dataset
    modelvis = vis.assign({"vis": xr.zeros_like(vis.vis)})

    # Can't directly pass dask arrays as map_blocks only loads xr.DataArrays
    if type(station_rm) == da.Array:
        # "id" is a coordinate from Configuration dataset
        station_rm = xr.DataArray(
            station_rm, coords={"id": np.arange(len(station_rm))}
        )

    # Call map_blocks function and return result
    return modelvis.map_blocks(
        _predict,
        args=[lsm, beam_type, eb_ms, eb_coeffs, station_rm, False],
        template=modelvis,
    )


def _get_beams(
    gainchunk: xr.Dataset,
    vischunk: xr.Dataset,
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
) -> xr.Dataset:
    """Return beam models used in prediction at beam centre.

    Set up to run with function generate_central_beams.

    :param gaintable: GainTable dataset containing initial solutions.
    :param vischunk: Visibility dataset containing observed data.

    :return: Chunked GainTable dataset
    """
    if len(vischunk.frequency) > 0:
        if np.any(gainchunk.frequency.data != vischunk.frequency.data):
            raise ValueError("Inconsistent frequencies")
        # Switch to standard variable names and coords for the SDP call
        gainchunk = gainchunk.rename({"soln_time": "time"})
        vischunk = restore_baselines_dim(vischunk)
        # Call solver
        generate_central_beams(
            gaintable=gainchunk,
            vis=vischunk,
            beam_type=beam_type,
            eb_coeffs=eb_coeffs,
            eb_ms=eb_ms,
        )
        # Change the time dimension name back for map_blocks I/O checks
        gainchunk = gainchunk.rename({"time": "soln_time"})

    return gainchunk


def prediction_central_beams(
    vis: xr.Dataset,
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
) -> xr.Dataset:
    """Return beam models used in prediction at beam centre.

    :param vis: Chunked Visibility dataset containing observed data.

    :return: Chunked GainTable dataset
    """
    # Create a bandpass calibration gain table
    fchunk = vis.chunks["frequency"][0]
    if fchunk <= 0:
        logger.warning("vis dataset does not appear to be chunked")
        fchunk = len(vis.frequency)
    gaintable = create_bandpass_table(vis).chunk({"frequency": fchunk})

    if len(gaintable.time) != 1:
        raise ValueError("error setting up gaintable")

    # map_blocks won't accept dimensions that differ but have the same name
    # So rename the gain time dimension (and coordinate)
    gaintable = gaintable.rename({"time": "soln_time"})
    gaintable = gaintable.map_blocks(
        _get_beams, args=[vis, beam_type, eb_ms, eb_coeffs]
    )

    # Undo any temporary variable name changes
    gaintable = gaintable.rename({"soln_time": "time"})
    return gaintable


def _solve(
    gainchunk: xr.Dataset,
    vischunk: xr.Dataset,
    modelchunk: xr.Dataset,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    phase_only: bool = False,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
    jones_type: Literal["T", "G", "B"] = "T",
    timeslice: float = None,
) -> xr.Dataset:
    """Call solve_bandpass.

    Set up to run with function run_solver.

    :param gaintable: GainTable dataset containing initial solutions.
    :param vischunk: Visibility dataset containing observed data.
    :param modelchunk: Visibility dataset containing model data.
    :param solver: Solver type to use. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0).
    :param niter: Number of solver iterations (defaults to 200).
    :param phase_only: Solve only for the phases.
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").
    :param jones_type: Type of calibration matrix T or G or B.
    :param timeslice: Defines the time scale over which each
        gain solution is valid.

    :return: Chunked GainTable dataset
    """
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
            phase_only=phase_only,
            tol=tol,
            crosspol=crosspol,
            normalise_gains=normalise_gains,
            jones_type=jones_type,
            timeslice=timeslice,
        )
        # Change the time dimension name back for map_blocks I/O checks
        gainchunk = gainchunk.rename({"time": "soln_time"})

    return gainchunk


def run_solver(
    vis: xr.Dataset,
    modelvis: xr.Dataset,
    gaintable: Optional[xr.Dataset] = None,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    phase_only: bool = False,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
    jones_type: Literal["T", "G", "B"] = "T",
    timeslice: float = None,
) -> xr.Dataset:
    """Do the bandpass calibration.

    :param vis: Chunked Visibility dataset containing observed data.
    :param modelvis: Chunked Visibility dataset containing model data.
    :param gaintable: Optional chunked GainTable dataset containing initial
        solutions.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0). Note that how referencing
        is done depends on the solver.
    :param niter: Number of solver iterations (defaults to 200).
    :param phase_only: Solve only for the phases.
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").
    :param jones_type: Type of calibration matrix T or G or B.
    :param timeslice: Defines the time scale over which each
        gain solution is valid.

    :return: Chunked GainTable dataset
    """
    # Create a full-band bandpass calibration gain table
    if gaintable is None:
        fchunk = vis.chunks["frequency"][0]
        if fchunk <= 0:
            logger.warning("vis dataset does not appear to be chunked")
            fchunk = len(vis.frequency)
        gaintable = create_bandpass_table(vis).chunk({"frequency": fchunk})

    if len(gaintable.time) != 1:
        raise ValueError("error setting up gaintable")

    if refant is not None:
        if refant < 0 or refant >= len(gaintable.antenna):
            raise ValueError(f"invalid refant: {refant}")

    # map_blocks won't accept dimensions that differ but have the same name
    # So rename the gain time dimension (and coordinate)
    gaintable = gaintable.rename({"time": "soln_time"})
    gaintable = gaintable.map_blocks(
        _solve,
        args=[
            vis,
            modelvis,
            solver,
            refant,
            niter,
            phase_only,
            tol,
            crosspol,
            normalise_gains,
            jones_type,
            timeslice,
        ],
    )

    # Undo any temporary variable name changes
    gaintable = gaintable.rename({"soln_time": "time"})
    return gaintable


def _solve_with_vis_setup(
    gainchunk: xr.Dataset,
    ms_name: str,
    frequency: npt.NDArray[float],
    lsm: list[Component],
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
    station_rm: Optional[npt.NDArray[float]] = None,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    phase_only: bool = False,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
    jones_type: Literal["T", "G", "B"] = "T",
    timeslice: Union[float, Literal["auto"], None] = None,
) -> xr.Dataset:
    """Call solve_bandpass.

    Set up to run with function run_solver.

    :param gainchunk: GainTable dataset containing initial solutions.
    :param ms_name: Name of input Measurement Set.
    :param frequency: list of all frequencies in the MSv2 dataset.
    :param lsm: List of LSM components. This is an intermediate format between
        the GSM and the evaluated SkyComponent list.
    :param beam_type: Type of beam model to use. Default is "everybeam".
    :param eb_ms: Pathname of Everybeam mock Measurement Set.
    :param eb_coeffs: Path to Everybeam coeffs directory.
    :param station_rm: Station rotation measure values. Default is None.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0). Note that how referencing
        is done depends on the solver.
    :param niter: Number of solver iterations (defaults to 200).
    :param phase_only: Solve only for the phases.
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").
    :param jones_type: Type of calibration matrix T or G or B.
    :param timeslice: Defines the time scale over which each
        gain solution is valid.

    :return: Chunked GainTable dataset
    """
    if len(gainchunk.frequency) > 0:

        # Load vis data for current band
        start = np.where(frequency == gainchunk.frequency.data[0])[0][0]
        end = np.where(frequency == gainchunk.frequency.data[-1])[0][0]
        vischunk = create_visibility_from_ms(
            ms_name, start_chan=start, end_chan=end
        )[0]

        # Fixme: remove reassignment once YAN-1990 is finalised
        #   current version sets vis: complex128, weight: float64, flags: int64
        vischunk = vischunk.assign(
            {
                "vis": vischunk.vis.astype(np.complex64),
                "weight": vischunk.weight.astype(np.float32),
                "flags": vischunk.flags.astype(bool),
            }
        )

        # Evaluate LSM for current band
        lsm_components = convert_model_to_skycomponents(
            lsm, vischunk.frequency.data
        )

        # Call predict
        modelchunk = vischunk.assign({"vis": xr.zeros_like(vischunk.vis)})
        predict_from_components(
            modelchunk,
            lsm_components,
            beam_type=beam_type,
            eb_coeffs=eb_coeffs,
            eb_ms=eb_ms,
            station_rm=station_rm,
        )
        # Fixme: remove reassignment once YAN-1990 is finalised
        #   current version sets vis: complex128, weight: float64, flags: int64
        modelchunk = modelchunk.assign(
            {
                "vis": modelchunk.vis.astype(np.complex64),
                "weight": modelchunk.weight.astype(np.float32),
                "flags": modelchunk.flags.astype(bool),
            }
        )

        # Call solver
        solve_bandpass(
            vis=vischunk,
            modelvis=modelchunk,
            gain_table=gainchunk,
            solver=solver,
            refant=refant,
            niter=niter,
            phase_only=phase_only,
            tol=tol,
            crosspol=crosspol,
            normalise_gains=normalise_gains,
            jones_type=jones_type,
            timeslice=timeslice,
        )

    return gainchunk


def ingest_predict_and_solve(
    ms_name: str,
    fchunk: int,
    lsm: list,
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
    gaintable: Optional[xr.Dataset] = None,
    station_rm: Optional[npt.NDArray[float]] = None,
    solver: str = "gain_substitution",
    refant: int = 0,
    niter: int = 200,
    phase_only: bool = False,
    tol: float = 1e-06,
    crosspol: bool = False,
    normalise_gains: str = None,
    jones_type: Literal["T", "G", "B"] = "T",
    timeslice: Union[float, Literal["auto"], None] = None,
) -> xr.Dataset:
    """Do the bandpass calibration, including initial ingest and predict.

    :param ms_name: Name of input Measurement Set.
    :param fchunk: Number of channels in the frequency chunks
    :param lsm: List of LSM components. This is an intermediate format between
        the GSM and the evaluated SkyComponent list.
    :param beam_type: Type of beam model to use. Default is "everybeam".
    :param eb_ms: Pathname of Everybeam mock Measurement Set.
    :param eb_coeffs: Path to Everybeam coeffs directory.
    :param station_rm: Station rotation measure values. Default is None.
    :param gaintable: Optional chunked GainTable dataset containing initial
        solutions.
    :param solver: Solver type to use. Currently any solver type accepted by
        solve_gaintable. Default is "gain_substitution".
    :param refant: Reference antenna (defaults to 0). Note that how referencing
        is done depends on the solver.
    :param niter: Number of solver iterations (defaults to 200).
    :param phase_only: Solve only for the phases.
    :param tol: Iteration stops when the fractional change in the gain solution
        is below this tolerance.
    :param crosspol: Do solutions including cross polarisations.
    :param normalise_gains: Normalises the gains (default="mean").
    :param jones_type: Type of calibration matrix T or G or B.
    :param timeslice: Defines the time scale over which each
        gain solution is valid.

    :return: Chunked GainTable dataset
    """
    # Get observation metadata
    ms_metadata = get_ms_metadata(ms_name)

    # Create a gain table if need be
    if gaintable is None:
        shape = (
            len(ms_metadata.time),
            len(ms_metadata.baselines),
            len(ms_metadata.frequency),
            len(
                PolarisationFrame.fits_codes[
                    ms_metadata.polarisation_frame.type
                ]
            ),
        )

        # Construct empty vis for create_bandpass_table
        vis = Visibility.constructor(
            uvw=ms_metadata.uvw,
            baselines=ms_metadata.baselines,
            time=ms_metadata.time,
            frequency=ms_metadata.frequency,
            channel_bandwidth=ms_metadata.channel_bandwidth,
            vis=da.zeros(shape, dtype=np.complex64),
            weight=da.zeros(shape, dtype=np.float32),
            flags=da.zeros(shape, dtype=bool),
            integration_time=ms_metadata.integration_time,
            configuration=ms_metadata.configuration,
            phasecentre=ms_metadata.phasecentre,
            polarisation_frame=ms_metadata.polarisation_frame,
            source=ms_metadata.source,
            meta=ms_metadata.meta,
            # Fixme: use new default once YAN-1990 is finalised
            low_precision="float32",
        )
        # Fixme: remove reassignment once YAN-1990 is finalised
        #   current version sets vis: input, weight: low_precision, flags: int64
        vis = vis.assign({"flags": xr.zeros_like(vis.flags, dtype=bool)})

        # Create a full-band bandpass calibration gain table
        gaintable = create_bandpass_table(vis).chunk({"frequency": fchunk})

        vis.close()

    if len(gaintable.time) != 1:
        raise ValueError("error setting up gaintable")

    if refant is not None:
        if refant < 0 or refant >= len(gaintable.antenna):
            raise ValueError(f"invalid refant: {refant}")

    # So rename the gain time dimension (and coordinate)
    gaintable = gaintable.map_blocks(
        _solve_with_vis_setup,
        args=[
            ms_name,
            ms_metadata.frequency,
            lsm,
            beam_type,
            eb_ms,
            eb_coeffs,
            station_rm,
            solver,
            refant,
            niter,
            phase_only,
            tol,
            crosspol,
            normalise_gains,
            jones_type,
            timeslice,
        ],
    )
    return gaintable


def _apply(
    vischunk: xr.Dataset,
    gainchunk: xr.Dataset,
    inverse: bool,
) -> xr.Dataset:
    """Call apply_gaintable.

    Set up to run with function apply_gaintable_to_dataset.

    :param vis: Visibility dataset to receive calibration factors.
    :param gaintable: GainTable dataset containing solutions to apply.
    :param inverse: Whether or not to apply the inverse.
    :return: Calibrated Visibility dataset
    """
    if len(vischunk.frequency) > 0:
        if np.any(gainchunk.frequency.data != vischunk.frequency.data):
            raise ValueError("Inconsistent frequencies")
        # Switch back to standard variable names for the SDP call
        gainchunk = gainchunk.rename({"soln_time": "time"})
        # Call apply function
        vischunk = apply_gaintable(vis=vischunk, gt=gainchunk, inverse=inverse)
    return vischunk


def apply_gaintable_to_dataset(
    vis: xr.Dataset,
    gaintable: xr.Dataset,
    inverse: bool = False,
) -> xr.Dataset:
    """Do the bandpass calibration.

    :param vis: Chunked Visibility dataset to receive calibration factors.
    :param gaintable: Chunked GainTable dataset containing solutions to apply.
    :param inverse: Apply the inverse. This requires the gain matrices to be
        square. (default=False)
    :return: Calibrated Visibility dataset
    """
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
            "baselineid"
        )
