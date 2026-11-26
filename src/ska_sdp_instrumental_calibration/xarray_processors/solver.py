import logging

import numpy as np
import xarray as xr

from ..data_managers.processing_set import (
    get_baseline_antenna_indices,
    get_correlated_data,
    get_data_group,
)
from ..data_managers.schema.calibration_set import GainCalibrationSetXds
from ..data_managers.schema.polarization import (
    CIRCULAR_POLS,
    LINEAR_POLS,
    jones_to_pol,
    pol_to_jones,
)
from ..numpy_processors.solvers import Solver

logger = logging.getLogger(__name__)

_TRANSPOSED_POL_ORDER = [0, 2, 1, 3]

_VIS_INPUTS = ("vis", "flags", "weight", "model_vis", "model_flags")

_GAIN_VARS = ("CALPARAM_GAIN", "CALPARAM_WEIGHT", "CALPARAM_RESIDUAL")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A bridge function between xarray.apply_ufunc and solver.solve.

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
    )
    return jones_to_pol(gain), jones_to_pol(weight), jones_to_pol(residual)


def _run_solver_ufunc_with_broadcast_frequency(
    *arrays: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    solver: Solver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solver wrapper for bandpass terms, parallelized across frequency.

    For Jones type B, frequency is kept out of the core dims so that dask
    distributes work over the visibility frequency chunks. ``apply_ufunc``
    then passes it as the first axis of every array, while the solver
    expects it just before the polarization axis: (time, baseline,
    frequency, pol) for visibilities, (time, antenna, frequency, pol) for
    gains and (time, frequency, pol) for residuals. It is moved there and
    back.

    Parameters
    ----------
    arrays
        Visibility and gain arrays, in the order of
        :func:`_run_solver_ufunc`, with frequency first.
    antenna1, antenna2, solver
        See :func:`_run_solver_ufunc`.

    Returns
    -------
        Gain, weight and residual arrays, with frequency first.
    """
    outputs = _run_solver_ufunc(
        *(np.moveaxis(array, 0, -2) for array in arrays),
        antenna1=antenna1,
        antenna2=antenna2,
        solver=solver,
    )
    return tuple(np.moveaxis(output, -2, 0) for output in outputs)


def _solver_inputs(
    vis: xr.DataTree, modelvis: xr.DataTree
) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """
    Solver inputs of observed and model MSv4 nodes, with every baseline
    ordered as antenna1 <= antenna2.

    The solvers were written for, and validated with, the MSv2 Visibility,
    whose reader orders baselines as antenna1 <= antenna2: baselines stored
    the other way are conjugated, with their cross-hand polarizations
    swapped (V21 = V12^H). MSv4 keeps the baselines as stored, and
    ``gain_substitution`` with cross polarizations is not invariant to the
    baseline order (it fills V21 from V12 without swapping cross hands). So
    the same reordering is applied here, lazily and only to reversed
    baselines, so that solutions are those of the MSv2 Visibility.

    Parameters
    ----------
    vis
        MSv4 node with observed correlated data, flags and weights, holding
        a single data group.
    modelvis
        MSv4 node with model correlated data and flags, on the same
        coordinates as ``vis``.

    Returns
    -------
        Dataset with the variables of :data:`_VIS_INPUTS`, and the
        antenna1 and antenna2 index of each baseline.

    Raises
    ------
    ValueError
        If there are reversed baselines and the polarizations are not
        XX XY YX YY or RR RL LR LL.
    """
    vis_group = get_data_group(vis)
    model_group = get_data_group(modelvis)
    inputs = xr.merge(
        [
            get_correlated_data(vis).rename("vis"),
            vis.ds[vis_group["flag"]].rename("flags"),
            vis.ds[vis_group["weight"]].rename("weight"),
            get_correlated_data(modelvis).rename("model_vis"),
            modelvis.ds[model_group["flag"]].rename("model_flags"),
        ],
        join="exact",
        compat="override",
        combine_attrs="drop",
    )
    antenna1, antenna2 = get_baseline_antenna_indices(vis)

    reversed_baselines = antenna1 > antenna2
    if reversed_baselines.any():
        inputs = _swap_baseline_antennas(inputs, reversed_baselines)
    return (
        inputs,
        np.minimum(antenna1, antenna2),
        np.maximum(antenna1, antenna2),
    )


def _swap_baseline_antennas(
    inputs: xr.Dataset, reversed_baselines: np.ndarray
) -> xr.Dataset:
    """
    Solver inputs of the reversed baselines as if stored as antenna2,
    antenna1: cross hands swapped, and visibilities conjugated.

    Raises
    ------
    ValueError
        If the polarizations are not XX XY YX YY or RR RL LR LL.
    """
    polarization = tuple(np.asarray(inputs.polarization, dtype=str))
    if polarization not in (LINEAR_POLS, CIRCULAR_POLS):
        raise ValueError(
            "Baselines with antenna1 > antenna2 are only supported for "
            f"polarizations {LINEAR_POLS} or {CIRCULAR_POLS}, got "
            f"{list(polarization)}"
        )
    swapped = inputs.isel(polarization=_TRANSPOSED_POL_ORDER).assign_coords(
        polarization=inputs.polarization
    )
    swapped["vis"] = np.conj(swapped.vis)
    swapped["model_vis"] = np.conj(swapped.model_vis)

    is_reversed = xr.DataArray(reversed_baselines, dims="baseline_id")
    return xr.where(is_reversed, swapped, inputs)


def run_solver(
    vis: xr.DataTree,
    modelvis: xr.DataTree,
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
        MSv4 node containing observed data, holding a single data group. If
        its backed by a dask array, then it can be chunked in time and
        frequency axis.
    modelvis
        MSv4 node containing model data, having similar shape, dtype and
        chunksizes as ``vis``, e.g. from
        :func:`~.predict.predict_vis`.
    gaintable
        GainCalibrationSetXds containing initial solutions, created from
        ``vis`` (same antennas, in ``antenna_xds`` order).
    solver
        An instance of solver, whose ``.solve()``
        method will be called, wrapped in :py:func:`xarray.apply_ufunc`
        for distributions across dask chunks

    Returns
    -------
        A new gaintable

    Raises
    ------
    ValueError
        If the gaintable antennas differ from those of ``vis``.
    """
    if not np.array_equal(
        gaintable.antenna_name.values,
        vis["antenna_xds"].ds.antenna_name.values,
    ):
        raise ValueError(
            "Gaintable antennas differ from the MSv4 antenna_xds antennas"
        )

    inputs, antenna1, antenna2 = _solver_inputs(vis, modelvis)

    vis_chunks_per_solution = {"time": -1}
    gaintable = gaintable.rename(time="solution_time")

    if gaintable.jones_type == "B":
        assert gaintable.frequency.size == inputs.frequency.size, (
            "For gaintable of type B, gaintable frequency size "
            "must match visibility frequency size"
        )
        # Chunk the gaintable such that it matches
        # visibility chunks across frequency dimension.
        # Chunks are read from the data only, as DataArray.chunksizes also
        # checks time coordinates, which may be chunked differently.
        vis_chunks = dict(zip(inputs.vis.dims, inputs.vis.chunks or ()))
        if vis_freq_chunks := vis_chunks.get("frequency", None):
            gaintable = gaintable.chunk(frequency=vis_freq_chunks)

        solver_ufunc = _run_solver_ufunc_with_broadcast_frequency
        vis_frequency, gain_frequency = [], []
    else:
        assert gaintable.frequency.size == 1, (
            "For gaintable of type T or G, "
            "gaintable frequency must be of size 1"
        )
        gaintable = gaintable.rename(frequency="solution_frequency")
        # Need to pass full frequency to process single solution
        vis_chunks_per_solution["frequency"] = -1
        solver_ufunc = _run_solver_ufunc
        vis_frequency, gain_frequency = ["frequency"], ["solution_frequency"]

    vis_core_dims = ["time", "baseline_id", *vis_frequency, "polarization"]
    gain_core_dims = [
        "solution_time",
        "antenna_name",
        *gain_frequency,
        "polarization",
    ]
    gain_vars_core_dims = [
        gain_core_dims,
        gain_core_dims,
        ["solution_time", *gain_frequency, "polarization"],
    ]

    gaintable_across_solutions = []
    for idx, slc in enumerate(gaintable.soln_interval_slices):
        solution_inputs = inputs.isel(time=slc).chunk(vis_chunks_per_solution)
        template_gaintable = gaintable.isel(solution_time=[idx])
        solutions = xr.apply_ufunc(
            solver_ufunc,
            *(solution_inputs[name] for name in _VIS_INPUTS),
            *(template_gaintable[name] for name in _GAIN_VARS),
            input_core_dims=[vis_core_dims] * len(_VIS_INPUTS)
            + gain_vars_core_dims,
            output_core_dims=gain_vars_core_dims,
            dask="parallelized",
            output_dtypes=[gaintable[name].dtype for name in _GAIN_VARS],
            kwargs={
                "antenna1": antenna1,
                "antenna2": antenna2,
                "solver": solver,
            },
        )
        gaintable_across_solutions.append(
            template_gaintable.assign(
                {
                    name: solution.transpose(*template_gaintable[name].dims)
                    for name, solution in zip(_GAIN_VARS, solutions)
                }
            )
        )

    combined_gaintable: GainCalibrationSetXds = xr.concat(
        gaintable_across_solutions, dim="solution_time"
    )

    combined_gaintable = combined_gaintable.rename(solution_time="time")

    if "solution_frequency" in combined_gaintable.dims:
        combined_gaintable = combined_gaintable.rename(
            solution_frequency="frequency"
        )

    return combined_gaintable
