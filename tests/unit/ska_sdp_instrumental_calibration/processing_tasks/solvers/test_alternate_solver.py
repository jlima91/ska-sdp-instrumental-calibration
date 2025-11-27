from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.processing_tasks.solvers import (
    alternative_solvers,
)

AlternativeSolver = alternative_solvers.AlternativeSolver
JonesSubtitution = alternative_solvers.JonesSubtitution
NormalEquation = alternative_solvers.NormalEquation
NormalEquationsPreSum = alternative_solvers.NormalEquationsPreSum


class TestAlternativeSolver:

    def test_alternative_solver_initialization(self):
        solver = AlternativeSolver()

        assert solver._solver_fn is None
        assert solver.niter == 50
        assert solver.tol == 1e-6

    def test_alternative_solver_initialization_with_params(self):
        solver = AlternativeSolver(niter=100, tol=1e-8)

        assert solver._solver_fn is None
        assert solver.niter == 100
        assert solver.tol == 1e-8

    def test_solve_raises_error_when_solver_fn_is_none(
        self, generate_vis_mvis_gain_ndarray_data
    ):
        mock_data = generate_vis_mvis_gain_ndarray_data

        solver = AlternativeSolver()

        with pytest.raises(ValueError) as err:
            solver.solve(
                vis_vis=mock_data["vis_vis"],
                vis_flags=mock_data["vis_flags"],
                vis_weight=mock_data["vis_weight"],
                model_vis=mock_data["model_vis"],
                model_flags=mock_data["model_flags"],
                gain_gain=mock_data["gain_gain"],
                gain_weight=mock_data["gain_weight"],
                gain_residual=mock_data["gain_residual"],
                ant1=mock_data["ant1"],
                ant2=mock_data["ant2"],
            )
        expected_message = (
            "AlternativeSolver: alternative solver "
            "function to be used is not provided."
        )
        assert str(err.value) == expected_message

    def test_solve_passes_correct_parameters_to_solver_fn(
        self, generate_vis_mvis_gain_ndarray_data
    ):
        mock_data = generate_vis_mvis_gain_ndarray_data
        mock_solver_fn = MagicMock()

        niter = 42
        tol = 1e-9

        solver = AlternativeSolver(niter=niter, tol=tol)
        solver._solver_fn = mock_solver_fn

        gain, weight, residual = solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )

        call_args = mock_solver_fn.call_args_list[0].args

        assert mock_solver_fn.called
        assert mock_solver_fn.call_count == mock_data["nchannels"]

        assert len(call_args) == 9

        np.testing.assert_equal(call_args[3], mock_data["ant1"])
        np.testing.assert_equal(call_args[4], mock_data["ant2"])
        np.testing.assert_equal(call_args[5], mock_data["gain_gain"][0])

        assert call_args[6] == 0
        assert call_args[7] == niter
        assert call_args[8] == tol

        np.testing.assert_equal(gain, mock_data["gain_gain"])
        np.testing.assert_equal(weight, mock_data["gain_weight"])
        np.testing.assert_equal(residual, mock_data["gain_residual"])


class TestJonesSubtitution:

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks.solvers."
        "alternative_solvers._jones_sub_solve"
    )
    def test_jones_substitution_initialization(self, jones_sub_solve_mock):
        solver = JonesSubtitution()

        assert solver._solver_fn is jones_sub_solve_mock
        assert solver.niter == 50
        assert solver.tol == 1e-6

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks.solvers."
        "alternative_solvers._jones_sub_solve"
    )
    def test_jones_substitution_solve(
        self, mock_jones_solve, generate_vis_mvis_gain_ndarray_data
    ):
        mock_data = generate_vis_mvis_gain_ndarray_data

        solver = JonesSubtitution(niter=25, tol=1e-5)

        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )

        assert mock_jones_solve.call_count == mock_data["nchannels"]


class TestNormalEquation:

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks.solvers."
        "alternative_solvers._normal_equation_solve"
    )
    def test_normal_equation_initialization(self, normal_equation_solve_mock):
        solver = NormalEquation()

        assert solver._solver_fn is normal_equation_solve_mock
        assert solver.niter == 50
        assert solver.tol == 1e-6

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks.solvers."
        "alternative_solvers._normal_equation_solve"
    )
    def test_normal_equation_solve(
        self, mock_normal_solve, generate_vis_mvis_gain_ndarray_data
    ):
        mock_data = generate_vis_mvis_gain_ndarray_data

        solver = NormalEquation(niter=20, tol=1e-4)

        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )

        assert mock_normal_solve.call_count == mock_data["nchannels"]


class TestNormalEquationsPreSum:

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks.solvers."
        "alternative_solvers._normal_equation_solve_with_presumming"
    )
    def test_normal_equations_presum_initialization(self, mock_presum_solve):
        solver = NormalEquationsPreSum()

        assert solver._solver_fn is mock_presum_solve
        assert solver.niter == 50
        assert solver.tol == 1e-6

    @patch(
        "ska_sdp_instrumental_calibration.processing_tasks.solvers."
        "alternative_solvers._normal_equation_solve_with_presumming"
    )
    def test_normal_equations_presum_solve(
        self, mock_presum_solve, generate_vis_mvis_gain_ndarray_data
    ):
        mock_data = generate_vis_mvis_gain_ndarray_data

        solver = NormalEquationsPreSum(niter=15, tol=1e-3)

        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )

        assert mock_presum_solve.call_count == mock_data["nchannels"]


def test_should_gains_without_normalising():

    solver = JonesSubtitution(normalise_gains="mean")
    gain = xr.DataArray(np.array([0, 1, 2, 3, 4]), dims=["frequency"])

    np.testing.assert_allclose(
        solver.normalise_gains(gain), np.array([0, 1, 2, 3, 4])
    )

    solver = NormalEquation(normalise_gains="mean")

    np.testing.assert_allclose(
        solver.normalise_gains(gain), np.array([0, 1, 2, 3, 4])
    )

    solver = NormalEquationsPreSum(normalise_gains="mean")

    np.testing.assert_allclose(
        solver.normalise_gains(gain), np.array([0, 1, 2, 3, 4])
    )
