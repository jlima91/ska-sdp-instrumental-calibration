"""
Unit tests for ska_sdp_instrumental_calibration.xarray_processors.solver
"""

from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_msv4,
)
from ska_sdp_instrumental_calibration.data_managers.processing_set import (
    with_correlated_data,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers import Solver
from ska_sdp_instrumental_calibration.xarray_processors.apply import (
    apply_gaintable_to_dataset,
)
from ska_sdp_instrumental_calibration.xarray_processors.solver import (
    _run_solver_ufunc,
    _run_solver_ufunc_with_broadcast_frequency,
    _solver_inputs,
    run_solver,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_NTIME, _NBL, _NFREQ, _NPOL = 3, 10, 8, 4
_NANT, _R1, _R2 = 5, 2, 2


def _freq_first_arrays():
    """Return numpy arrays in frequency-first
    order (as xarray broadcast sends)."""
    vis_data = np.ones((_NFREQ, _NTIME, _NBL, _NPOL), dtype=np.complex64)
    vis_vis = vis_data.copy()
    vis_flags = vis_data.copy().astype(bool)
    vis_weight = np.ones(vis_data.shape, dtype=np.float32)
    model_vis = 2 * vis_data.copy()
    model_flags = vis_data.copy().astype(bool)

    gain_data = np.ones((_NFREQ, _NTIME, _NANT, _NPOL), dtype=np.complex64)
    gain_gain = gain_data.copy()
    gain_weight = np.ones(gain_data.shape, dtype=np.float32)

    gain_residual = np.ones((_NFREQ, _NTIME, _NPOL), dtype=np.float32)

    antenna1 = np.arange(_NBL)
    antenna2 = np.arange(_NBL)

    return (
        vis_vis,
        vis_flags,
        vis_weight,
        model_vis,
        model_flags,
        gain_gain,
        gain_weight,
        gain_residual,
        antenna1,
        antenna2,
    )


def _make_solver_returning_time_first():
    """Return mock solver whose .solve returns
    arrays in solver-expected order."""
    solver = MagicMock()
    solver.solve.return_value = (
        np.ones((_NTIME, _NANT, _NFREQ, _R1, _R2), dtype=np.complex64),
        np.ones((_NTIME, _NANT, _NFREQ, _R1, _R2), dtype=np.float32),
        np.ones((_NTIME, _NFREQ, _R1, _R2), dtype=np.float32),
    )
    return solver


def _apply_ufunc_return_template_arrays(*args, **kwargs):
    """Side-effect: return template gain/weight/residual unchanged.

    apply_ufunc positional args order:
      0: ufunc, 1: vis.vis, 2: vis.flags, 3: vis.weight,
      4: model.vis, 5: model.flags,
      6: gain, 7: weight, 8: residual
    """
    return args[6], args[7], args[8]


class TestRunSolverUfunc:

    def _arrays(self):
        return (
            "vis_vis",
            "vis_flags",
            "vis_weight",
            "model_vis",
            "model_flags",
            "CALPARAM_GAIN",
            "CALPARAM_WEIGHT",
            "CALPARAM_RESIDUAL",
            "antenna1",
            "antenna2",
        )

    @patch(
        "ska_sdp_instrumental_calibration.xarray_processors"
        ".solver.jones_to_pol",
        new=lambda x: x,
    )
    @patch(
        "ska_sdp_instrumental_calibration.xarray_processors"
        ".solver.pol_to_jones",
        new=lambda x: x,
    )
    def test_calls_solver_solve_with_all_args(self):
        """_run_solver_ufunc delegates directly to solver.solve."""
        solver = MagicMock()
        expected = (MagicMock(), MagicMock(), MagicMock())
        solver.solve.return_value = expected
        arrays = self._arrays()

        result = _run_solver_ufunc(*arrays, solver)

        solver.solve.assert_called_once_with(*arrays)
        assert result == expected


class TestRunSolverUfuncWithBroadcastFrequency:
    """
    Verifies that _run_solver_ufunc_with_broadcast_frequency correctly
    transposes freq-first arrays to solver-expected order before calling
    solver.solve, and transposes outputs back to freq-first.
    """

    def test_arrays_transposed_to_order_expected_by_solver(self):
        """solver.solve receives vis with freq moved from dim-0 to dim-2."""
        *arrays, antenna1, antenna2 = _freq_first_arrays()
        solver = _make_solver_returning_time_first()

        gain_out, weight_out, residual_out = (
            _run_solver_ufunc_with_broadcast_frequency(
                *arrays, antenna1=antenna1, antenna2=antenna2, solver=solver
            )
        )

        solve_args = solver.solve.call_args.args
        # Expected shape: (time, baseline, freq, pol)
        expected_vis_shape = (_NTIME, _NBL, _NFREQ, _NPOL)
        assert solve_args[0].shape == expected_vis_shape  # vis_vis
        assert solve_args[1].shape == expected_vis_shape  # vis_flags
        assert solve_args[2].shape == expected_vis_shape  # vis_weight
        assert solve_args[3].shape == expected_vis_shape  # model_vis
        assert solve_args[4].shape == expected_vis_shape  # model_flags

        expected_gain_shape = (_NTIME, _NANT, _NFREQ, _R1, _R2)
        assert solve_args[5].shape == expected_gain_shape  # gain_gain
        assert solve_args[6].shape == expected_gain_shape  # gain_weight

        assert solve_args[7].shape == (
            _NTIME,
            _NFREQ,
            _R1,
            _R2,
        )  # gain_residual

        # Output shape has frequency dimension as first
        assert gain_out.shape == (_NFREQ, _NTIME, _NANT, _NPOL)
        assert weight_out.shape == (_NFREQ, _NTIME, _NANT, _NPOL)
        assert residual_out.shape == (_NFREQ, _NTIME, _NPOL)


# ── run_solver on MSv4 nodes ─────────────────────────────────────────────────
# The conftest MSv4 node has 3 antennas and baselines (a0, a1), (a2, a1)
# (reversed, antenna1 > antenna2) and (a0, a2).
_REVERSED_BASELINE = 1
_TRUE_GAINS = np.array([1.0 + 0.0j, 1.2 - 0.3j, 0.8 + 0.5j])


def _corrupted_vis_and_model(ms_xdt):
    """
    Model node with the conftest visibilities, and a vis node corrupted
    by diagonal antenna gains: V = g1 M g2^H.
    """
    model = ms_xdt.ds.VISIBILITY
    antenna1, antenna2 = np.array([0, 2, 0]), np.array([1, 1, 2])
    baseline_gain = _TRUE_GAINS[antenna1] * np.conj(_TRUE_GAINS[antenna2])
    corrupted = model * xr.DataArray(baseline_gain, dims="baseline_id")
    return (
        with_correlated_data(ms_xdt, corrupted.astype(model.dtype)),
        ms_xdt,
    )


def _canonical(ms_xdt):
    """
    The same node, with the reversed baseline stored as antenna1 <
    antenna2: names swapped, visibilities conjugated with cross hands
    swapped, flags and weights with cross hands swapped.
    """
    ds = ms_xdt.to_dataset()
    rev = xr.DataArray(
        np.arange(ds.baseline_id.size) == _REVERSED_BASELINE,
        dims="baseline_id",
    )

    def swap_cross_hands(data):
        return data.isel(polarization=[0, 2, 1, 3]).assign_coords(
            polarization=data.polarization
        )

    canonical = ds.assign(
        VISIBILITY=xr.where(
            rev, np.conj(swap_cross_hands(ds.VISIBILITY)), ds.VISIBILITY
        ),
        FLAG=xr.where(rev, swap_cross_hands(ds.FLAG), ds.FLAG),
        WEIGHT=xr.where(rev, swap_cross_hands(ds.WEIGHT), ds.WEIGHT),
    ).assign_coords(
        baseline_antenna1_name=xr.where(
            rev, ds.baseline_antenna2_name, ds.baseline_antenna1_name
        ),
        baseline_antenna2_name=xr.where(
            rev, ds.baseline_antenna1_name, ds.baseline_antenna2_name
        ),
    )
    node = ms_xdt.copy(deep=False)
    node.ds = canonical.assign_attrs(ds.attrs)
    return node


class TestRunSolverMSv4:

    @pytest.mark.parametrize("jones_type", ["B", "G"])
    def test_solved_gains_reproduce_corrupted_visibilities(
        self, make_ms_xdt, jones_type
    ):
        vis, modelvis = _corrupted_vis_and_model(make_ms_xdt())
        gaintable = create_gaintable_from_msv4(
            vis, "full", jones_type, lower_precision=False
        )
        solver = Solver.get_solver(refant=0, niter=200, tol=1e-12)

        solved = run_solver(vis, modelvis, gaintable, solver)

        assert isinstance(solved.CALPARAM_GAIN.data, da.Array)
        assert solved.CALPARAM_GAIN.dims == gaintable.CALPARAM_GAIN.dims
        np.testing.assert_array_equal(solved.time, gaintable.time)
        # Gains are only defined up to a common phase, so compare the
        # visibilities they predict rather than the gains themselves
        np.testing.assert_allclose(
            apply_gaintable_to_dataset(modelvis, solved).ds.VISIBILITY,
            vis.ds.VISIBILITY,
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.parametrize("crosspol", [False, True])
    def test_reversed_baselines_solve_as_antenna1_before_antenna2(
        self, make_ms_xdt, crosspol
    ):
        vis, modelvis = _corrupted_vis_and_model(make_ms_xdt())
        gaintable = create_gaintable_from_msv4(vis, "full", "B")
        solver = Solver.get_solver(refant=0, niter=50, crosspol=crosspol)

        solved = run_solver(vis, modelvis, gaintable, solver)
        expected = run_solver(
            _canonical(vis), _canonical(modelvis), gaintable, solver
        )

        for name in ("CALPARAM_GAIN", "CALPARAM_WEIGHT", "CALPARAM_RESIDUAL"):
            np.testing.assert_array_equal(
                solved[name].values, expected[name].values, err_msg=name
            )

    def test_should_pass_ordered_baselines_unchanged(self, make_ms_xdt):
        canonical = _canonical(make_ms_xdt())

        inputs, antenna1, antenna2 = _solver_inputs(canonical, canonical)

        assert inputs.vis.data is canonical.ds.VISIBILITY.data
        assert inputs.flags.data is canonical.ds.FLAG.data
        assert inputs.weight.data is canonical.ds.WEIGHT.data
        assert inputs.model_vis.data is canonical.ds.VISIBILITY.data
        assert inputs.model_flags.data is canonical.ds.FLAG.data
        np.testing.assert_array_equal(antenna1, [0, 1, 0])
        np.testing.assert_array_equal(antenna2, [1, 2, 2])

    def test_should_raise_for_other_antenna_order(self, make_ms_xdt):
        ms_xdt = make_ms_xdt()
        gaintable = create_gaintable_from_msv4(ms_xdt, "full", "B")

        with pytest.raises(ValueError, match="antennas differ"):
            run_solver(
                ms_xdt,
                ms_xdt,
                gaintable.isel(antenna_name=[1, 0, 2]),
                Solver.get_solver(),
            )

    def test_should_raise_reordering_unsupported_polarizations(
        self, make_ms_xdt
    ):
        ms_xdt = make_ms_xdt(polarization=("XX", "YY", "XY", "YX"))

        with pytest.raises(ValueError, match="antenna1 > antenna2"):
            _solver_inputs(ms_xdt, ms_xdt)

    def test_should_raise_for_model_on_other_coordinates(self, make_ms_xdt):
        vis = make_ms_xdt()
        modelvis = vis.copy(deep=False)
        modelvis.ds = vis.ds.isel(time=slice(1, None))

        with pytest.raises(ValueError, match="cannot align"):
            _solver_inputs(vis, modelvis)
