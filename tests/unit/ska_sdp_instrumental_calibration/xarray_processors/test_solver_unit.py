"""
Unit tests for ska_sdp_instrumental_calibration.xarray_processors.solver
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_visibility,
)
from ska_sdp_instrumental_calibration.xarray_processors.solver import (
    _run_solver_ufunc,
    _run_solver_ufunc_with_broadcast_frequency,
    run_solver,
)

_APPLY_UFUNC_PATCH = (
    "ska_sdp_instrumental_calibration.xarray_processors.solver.xr.apply_ufunc"
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
    vis_weight = vis_data.copy().astype(np.float32)
    model_vis = 2 * vis_data.copy()
    model_flags = vis_data.copy().astype(bool)

    gain_data = np.ones((_NFREQ, _NTIME, _NANT, _R1, _R2), dtype=np.complex64)
    gain_gain = gain_data.copy()
    gain_weight = gain_data.copy().astype(np.float32)

    gain_residual = np.ones((_NFREQ, _NTIME, _R1, _R2), dtype=np.float32)

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
            "gain",
            "gain_weight",
            "gain_residual",
            "antenna1",
            "antenna2",
        )

    def test_calls_solver_solve_with_all_args(self):
        """_run_solver_ufunc delegates directly to solver.solve."""
        solver = MagicMock()
        expected = (MagicMock(), MagicMock(), MagicMock())
        solver.solve.return_value = expected
        arrays = self._arrays()

        result = _run_solver_ufunc(*arrays, solver)

        solver.solve.assert_called_once_with(*arrays)
        assert result is expected


class TestRunSolverUfuncWithBroadcastFrequency:
    """
    Verifies that _run_solver_ufunc_with_broadcast_frequency correctly
    transposes freq-first arrays to solver-expected order before calling
    solver.solve, and transposes outputs back to freq-first.
    """

    def test_arrays_transposed_to_order_expected_by_solver(self):
        """solver.solve receives vis with freq moved from dim-0 to dim-2."""
        arrays = _freq_first_arrays()
        solver = _make_solver_returning_time_first()

        gain_out, weight_out, residual_out = (
            _run_solver_ufunc_with_broadcast_frequency(*arrays, solver)
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
        assert gain_out.shape == (_NFREQ, _NTIME, _NANT, _R1, _R2)
        assert weight_out.shape == (_NFREQ, _NTIME, _NANT, _R1, _R2)
        assert residual_out.shape == (_NFREQ, _NTIME, _R1, _R2)


class TestRunSolver:
    """run_solver tests with mocked xr.apply_ufunc."""

    def test_jones_B_uses_broadcast_frequency_ufunc(self, generate_vis):
        vis, gaintable = generate_vis
        modelvis = vis.copy(deep=True)
        solver_instance = MagicMock()

        with patch(
            _APPLY_UFUNC_PATCH,
            side_effect=_apply_ufunc_return_template_arrays,
        ) as mock_ufunc:
            result = run_solver(vis, modelvis, gaintable, solver_instance)

        # Fixture generate_vis creates gaintable with 3 solution intervals,
        # since visbility has 3 time slots
        # apply_ufunc will be called one per solution interval
        assert len(mock_ufunc.call_args_list) == len(
            gaintable.soln_interval_slices
        )

        for call in mock_ufunc.call_args_list:
            assert call.args[0] is _run_solver_ufunc_with_broadcast_frequency

        # args[6] is template_gaintable.gain passed to apply_ufunc
        # assert that gaintable is not chunked, since vis is not chunked
        gain_arg = mock_ufunc.call_args_list[0].args[6]
        assert "frequency" not in gain_arg.chunksizes

        # For B type, frequency is a broadcast dim — not in core dims.
        kw = mock_ufunc.call_args_list[0].kwargs
        assert kw["input_core_dims"][0] == [
            "time",
            "baselineid",
            "polarisation",
        ]
        assert kw["output_core_dims"][0] == [
            "solution_time",
            "antenna",
            "receptor1",
            "receptor2",
        ]
        assert kw["output_core_dims"][2] == [
            "solution_time",
            "receptor1",
            "receptor2",
        ]
        assert kw["dask"] == "parallelized"
        ufunc_kwargs = kw["kwargs"]
        assert ufunc_kwargs["solver"] is solver_instance
        np.testing.assert_array_equal(
            ufunc_kwargs["antenna1"], vis.antenna1.data
        )
        np.testing.assert_array_equal(
            ufunc_kwargs["antenna2"], vis.antenna2.data
        )

        assert "time" in result.dims
        assert "frequency" in result.dims
        assert "solution_time" not in result.dims
        assert "solution_frequency" not in result.dims

    def test_jones_G_uses_standard_ufunc(self, generate_vis):
        vis, _ = generate_vis
        gaintable = create_gaintable_from_visibility(
            vis, jones_type="G", timeslice="full", skip_default_chunk=True
        ).compute()
        modelvis = vis.copy(deep=True)
        solver_instance = MagicMock()

        with patch(
            _APPLY_UFUNC_PATCH,
            side_effect=_apply_ufunc_return_template_arrays,
        ) as mock_ufunc:
            result = run_solver(vis, modelvis, gaintable, solver_instance)

        # Since timeslice="full", gaintable only has 1 solution interval
        assert len(mock_ufunc.call_args_list) == 1
        assert mock_ufunc.call_args_list[0].args[0] is _run_solver_ufunc

        # args[6] is template_gaintable.gain passed to apply_ufunc
        # assert that gaintable is not chunked, since vis is not chunked
        gain_arg = mock_ufunc.call_args_list[0].args[6]
        assert "frequency" not in gain_arg.chunksizes

        # For G type, frequency is a core dim passed to the solver.
        kw = mock_ufunc.call_args_list[0].kwargs
        assert kw["input_core_dims"][0] == [
            "time",
            "baselineid",
            "frequency",
            "polarisation",
        ]
        assert kw["output_core_dims"][0] == [
            "solution_time",
            "antenna",
            "solution_frequency",
            "receptor1",
            "receptor2",
        ]
        assert kw["output_core_dims"][2] == [
            "solution_time",
            "solution_frequency",
            "receptor1",
            "receptor2",
        ]
        assert kw["dask"] == "parallelized"
        ufunc_kwargs = kw["kwargs"]
        assert ufunc_kwargs["solver"] is solver_instance
        np.testing.assert_array_equal(
            ufunc_kwargs["antenna1"], vis.antenna1.data
        )
        np.testing.assert_array_equal(
            ufunc_kwargs["antenna2"], vis.antenna2.data
        )

        assert "time" in result.dims
        assert "frequency" in result.dims
        assert "solution_time" not in result.dims
        assert "solution_frequency" not in result.dims

    def test_jones_B_rechunks_gaintable_to_match_vis_frequency_chunks(
        self, generate_vis
    ):
        """When vis is dask-chunked in frequency,
        gaintable is rechunked to match."""
        vis, gaintable = generate_vis
        freq_chunk_size = 3  # vis has 4 freqs → two chunks -> (3, 1)
        vis = vis.chunk({"frequency": freq_chunk_size})
        modelvis = vis.copy(deep=True)
        solver_instance = MagicMock()

        with patch(
            _APPLY_UFUNC_PATCH,
            side_effect=_apply_ufunc_return_template_arrays,
        ) as mock_ufunc:
            run_solver(vis, modelvis, gaintable, solver_instance)

        # args[6] is template_gaintable.gain passed to apply_ufunc
        gain_arg = mock_ufunc.call_args_list[0].args[6]
        expected_freq_chunks = (3, 1)
        assert gain_arg.chunksizes.get("frequency") == expected_freq_chunks

    def test_jones_B_raises_on_frequency_size_mismatch(self, generate_vis):
        """B-type gaintable must have same frequency count as visibility."""
        vis, _ = generate_vis  # vis has 4 freqs
        mock_gt = MagicMock()
        mock_gt.jones_type = "B"
        mock_gt.rename.return_value = mock_gt
        mock_gt.frequency.size = 2  # ≠ vis.frequency.size (4)
        mock_gt.soln_interval_slices = [slice(0, 3)]

        with pytest.raises(
            AssertionError,
            match=(
                "For gaintable of type B, gaintable frequency size "
                "must match visibility frequency size"
            ),
        ):
            run_solver(vis, vis.copy(deep=True), mock_gt, MagicMock())

    def test_jones_G_raises_if_frequency_size_not_one(self, generate_vis):
        """G/T-type gaintable must have exactly 1 frequency channel."""
        vis, _ = generate_vis
        mock_gt = MagicMock()
        mock_gt.jones_type = "G"
        mock_gt.rename.return_value = mock_gt
        mock_gt.frequency.size = 4  # ≠ 1
        mock_gt.soln_interval_slices = [slice(0, 3)]

        with pytest.raises(
            AssertionError,
            match=(
                "For gaintable of type T or G, "
                "gaintable frequency must be of size 1"
            ),
        ):
            run_solver(vis, vis.copy(deep=True), mock_gt, MagicMock())
