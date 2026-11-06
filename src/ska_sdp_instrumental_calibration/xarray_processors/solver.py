import dataclasses
import logging

import numpy as np
import xarray as xr
from ska_sdp_datamodels.visibility import Visibility

from ..data_managers.schema.calibration_set import GainCalibrationSetXds
from ..data_managers.schema.polarization import jones_to_pol, pol_to_jones
from ..numpy_processors.solvers import Solver
from ..numpy_processors.solvers.dp3_solvers import DP3ObservationInfo

logger = logging.getLogger(__name__)


def _run_solver_ufunc(
    vis_vis: np.ndarray,
    vis_flags: np.ndarray,
    vis_weight: np.ndarray,
    model_vis: np.ndarray,
    model_flags: np.ndarray,
    gain_gain: np.ndarray,
    gain_weight: np.ndarray,
    gain_residual: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    solver: Solver,
    **solve_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A bridge function between xarray.apply_ufunc and solver.solve.

    ``solve_kwargs`` (e.g. ``observation``) are passed on to
    ``solver.solve``.

    Returns
    -------
        Gain, weight and residual arrays returned by the solver.
    """
    gain, weight, residual = solver.solve(
        vis_vis,
        vis_flags,
        vis_weight,
        model_vis,
        model_flags,
        pol_to_jones(gain_gain),
        pol_to_jones(gain_weight),
        pol_to_jones(gain_residual),
        antenna1,
        antenna2,
        **solve_kwargs,
    )
    return jones_to_pol(gain), jones_to_pol(weight), jones_to_pol(residual)


def _run_solver_ufunc_with_broadcast_frequency(
    vis_vis: np.ndarray,
    vis_flags: np.ndarray,
    vis_weight: np.ndarray,
    model_vis: np.ndarray,
    model_flags: np.ndarray,
    gain_gain: np.ndarray,
    gain_weight: np.ndarray,
    gain_residual: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    solver: Solver,
    **solve_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solver wrapper for bandpass terms where computation is parallelized
    across frequency dimension, using xarray.apply_ufunc.

    Xarray passes broadcast dimensions ahead of core dimensions. For Jones
    type B, we keep frequency outside the core dims so dask can still
    distribute work over visibility frequency chunks.
    """
    # Order expected by solver:     time, baseline, frequency, polarisation
    # Order of input data:          frequency, time, baseline, polarisation
    # We need to re-order so that frequency becomes 3rd dimension
    vis_dim_order = (1, 2, 0, 3)

    # Order expected by solver:     time, antenna, frequency, pol
    # Order of input data:          frequency, time, antenna, pol
    # We need to re-order so that frequency becomes 3rd dimension
    gain_dim_order = (1, 2, 0, 3)

    # Order expected by solver:     time, frequency, pol
    # Order of input data:          frequency, time, pol
    # We need to re-order so that frequency becomes 2nd dimension
    residual_dim_order = (1, 0, 2)

    gain, weight, residual = _run_solver_ufunc(
        vis_vis=np.transpose(vis_vis, vis_dim_order),
        vis_flags=np.transpose(vis_flags, vis_dim_order),
        vis_weight=np.transpose(vis_weight, vis_dim_order),
        model_vis=np.transpose(model_vis, vis_dim_order),
        model_flags=np.transpose(model_flags, vis_dim_order),
        gain_gain=np.transpose(gain_gain, gain_dim_order),
        gain_weight=np.transpose(gain_weight, gain_dim_order),
        gain_residual=np.transpose(gain_residual, residual_dim_order),
        antenna1=antenna1,
        antenna2=antenna2,
        solver=solver,
        **solve_kwargs,
    )

    # Before returning, move frequency from 3rd to 1st dimension in order
    return (
        np.transpose(gain, (2, 0, 1, 3)),
        np.transpose(weight, (2, 0, 1, 3)),
        np.transpose(residual, (1, 0, 2)),
    )


def _run_solver_ufunc_with_observation(
    vis_vis: np.ndarray,
    vis_flags: np.ndarray,
    vis_weight: np.ndarray,
    model_vis: np.ndarray,
    model_flags: np.ndarray,
    gain_gain: np.ndarray,
    gain_weight: np.ndarray,
    gain_residual: np.ndarray,
    uvw: np.ndarray,
    frequency: np.ndarray,
    channel_bandwidth: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    solver: Solver,
    observation: DP3ObservationInfo,
    solver_ufunc,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bridge for solvers that need the observation metadata of the chunk
    (``solver.requires_observation``).

    ``observation`` holds the metadata shared by all chunks of a solution
    interval. UVW, frequency and channel bandwidth are passed as chunked
    inputs, as they differ per chunk; they are set on ``observation``
    before calling ``solver_ufunc`` (one of the bridges above), which
    passes it on to ``solver.solve``.

    Returns
    -------
        Gain, weight and residual arrays returned by the solver.
    """
    # Xarray inserts length-1 axes for broadcast dims that UVW lacks
    # (frequency, for Jones type B), so keep only (time, baseline, spatial)
    chunk_observation = dataclasses.replace(
        observation,
        uvw=uvw.reshape(uvw.shape[-3:]),
        frequency=np.ravel(frequency),
        channel_bandwidth=np.ravel(channel_bandwidth),
    )
    return solver_ufunc(
        vis_vis,
        vis_flags,
        vis_weight,
        model_vis,
        model_flags,
        gain_gain,
        gain_weight,
        gain_residual,
        antenna1=antenna1,
        antenna2=antenna2,
        solver=solver,
        observation=chunk_observation,
    )


def run_solver(
    vis: Visibility,
    modelvis: Visibility,
    gaintable: GainCalibrationSetXds,
    solver: Solver,
) -> GainCalibrationSetXds:
    """
    A function used for distributing the ``solver.solve()`` function
    call across solution intervals of gaintable, and across the chunks
    of visibility.

    Parameters
    ----------
    vis
        Visibility dataset containing observed data. If its backed by a dask
        array, then it can be chunked in time and frequency axis.
    modelvis
        Visibility dataset containing model data, having similar shape,
        dtype and chunksizes as ``vis``
    gaintable
        GainCalibrationSetXds containing initial solutions.
    solver
        An instance of solver, whose ``.solve()``
        method will be called, wrapped in :py:func:`xarray.apply_ufunc`
        for distributions across dask chunks

    Returns
    -------
        A new gaintable
    """

    # Solvers that predict their own model (e.g. dp3_ddecal) need the
    # observation metadata and UVW of each chunk
    requires_observation = (
        getattr(solver, "requires_observation", False) is True
    )

    vis_chunks_per_solution = {"time": -1}
    gaintable = gaintable.rename(time="solution_time")
    soln_interval_slices = gaintable.soln_interval_slices
    output_dtypes = [
        gaintable.CALPARAM_GAIN.dtype,
        gaintable.CALPARAM_WEIGHT.dtype,
        gaintable.CALPARAM_RESIDUAL.dtype,
    ]

    if gaintable.jones_type == "B":
        assert gaintable.frequency.size == vis.frequency.size, (
            "For gaintable of type B, gaintable frequency size "
            "must match visibility frequency size"
        )
        # Chunk the gaintable such that it matches
        # visibility chunks across frequency dimension
        if vis_freq_chunks := vis.chunksizes.get("frequency", None):
            gaintable = gaintable.chunk(frequency=vis_freq_chunks)

        solver_ufunc = _run_solver_ufunc_with_broadcast_frequency
        vis_core_dims = ["time", "baselineid", "polarisation"]
        gain_core_dims = [
            "solution_time",
            "antenna_name",
            "polarization",
        ]
        residual_core_dims = [
            "solution_time",
            "polarization",
        ]
    else:  # jones_type == T or G
        assert gaintable.frequency.size == 1, (
            "For gaintable of type T or G, "
            "gaintable frequency must be of size 1"
        )
        gaintable = gaintable.rename(frequency="solution_frequency")
        # Need to pass full frequency to process single solution
        vis_chunks_per_solution["frequency"] = -1
        solver_ufunc = _run_solver_ufunc
        vis_core_dims = ["time", "baselineid", "frequency", "polarisation"]
        gain_core_dims = [
            "solution_time",
            "antenna_name",
            "solution_frequency",
            "polarization",
        ]
        residual_core_dims = [
            "solution_time",
            "solution_frequency",
            "polarization",
        ]

    gaintable_across_solutions = []
    for idx, slc in enumerate(soln_interval_slices):
        vis_per_solution = vis.isel(time=slc).chunk(vis_chunks_per_solution)
        modelvis_per_solution = modelvis.isel(time=slc).chunk(
            vis_chunks_per_solution
        )
        template_gaintable = gaintable.isel(
            solution_time=[idx]
        )  # Select index but keep dimension

        inputs = [
            vis_per_solution.vis,
            vis_per_solution.flags,
            vis_per_solution.weight,
            modelvis_per_solution.vis,
            modelvis_per_solution.flags,
            template_gaintable.CALPARAM_GAIN,
            template_gaintable.CALPARAM_WEIGHT,
            template_gaintable.CALPARAM_RESIDUAL,
        ]
        input_core_dims = [
            vis_core_dims,
            vis_core_dims,
            vis_core_dims,
            vis_core_dims,
            vis_core_dims,
            gain_core_dims,
            gain_core_dims,
            residual_core_dims,
        ]
        ufunc_kwargs = {
            "antenna1": vis.antenna1.data,
            "antenna2": vis.antenna2.data,
            "solver": solver,
        }
        ufunc = solver_ufunc
        if requires_observation:
            # UVW, frequency and channel bandwidth differ per chunk, so they
            # go in as inputs chunked like the visibilities. Frequency is a
            # core dim for T/G and a broadcast dim for B.
            freq_core_dims = (
                ["frequency"] if "frequency" in vis_core_dims else []
            )
            freq_chunks = vis_per_solution.vis.chunksizes.get("frequency")
            for name in ("frequency", "channel_bandwidth"):
                coord = vis_per_solution[name].reset_coords(drop=True)
                inputs.append(
                    coord.chunk(frequency=freq_chunks)
                    if freq_chunks
                    else coord
                )
            inputs.insert(-2, vis_per_solution.uvw)
            input_core_dims += [
                ["time", "baselineid", "spatial"],
                freq_core_dims,
                freq_core_dims,
            ]
            ufunc_kwargs["observation"] = dataclasses.replace(
                DP3ObservationInfo.from_visibility(vis_per_solution),
                uvw=None,
                frequency=None,
                channel_bandwidth=None,
            )
            ufunc_kwargs["solver_ufunc"] = solver_ufunc
            ufunc = _run_solver_ufunc_with_observation

        gain, weight, residual = xr.apply_ufunc(
            ufunc,
            *inputs,
            input_core_dims=input_core_dims,
            output_core_dims=[
                gain_core_dims,
                gain_core_dims,
                residual_core_dims,
            ],
            dask="parallelized",
            output_dtypes=output_dtypes,
            kwargs=ufunc_kwargs,
        )
        gaintable_per_solution = template_gaintable.assign(
            CALPARAM_GAIN=gain.transpose(
                *template_gaintable.CALPARAM_GAIN.dims
            ),
            CALPARAM_WEIGHT=weight.transpose(
                *template_gaintable.CALPARAM_WEIGHT.dims
            ),
            CALPARAM_RESIDUAL=residual.transpose(
                *template_gaintable.CALPARAM_RESIDUAL.dims
            ),
        )
        gaintable_across_solutions.append(gaintable_per_solution)

    combined_gaintable: GainCalibrationSetXds = xr.concat(
        gaintable_across_solutions, dim="solution_time"
    )

    combined_gaintable = combined_gaintable.rename(solution_time="time")

    if "solution_frequency" in combined_gaintable.dims:
        combined_gaintable = combined_gaintable.rename(
            solution_frequency="frequency"
        )

    return combined_gaintable
