import pytest

from ska_sdp_instrumental_calibration.numpy_processors.solvers import (
    alternative_solvers,
    gain_substitution_solver,
    solver,
)

JonesSubtitution = alternative_solvers.JonesSubtitution
NormalEquation = alternative_solvers.NormalEquation
NormalEquationsPreSum = alternative_solvers.NormalEquationsPreSum
GainSubstitution = gain_substitution_solver.GainSubstitution
Solver = solver.Solver


def test_get_solver_returns_gain_substitution():
    solver = Solver.get_solver("gain_substitution")

    assert isinstance(solver, GainSubstitution)


def test_get_solver_returns_jones_substitution():
    solver = Solver.get_solver("jones_substitution")

    assert isinstance(solver, JonesSubtitution)


def test_get_solver_returns_normal_equations():
    solver = Solver.get_solver("normal_equations")

    assert isinstance(solver, NormalEquation)


def test_get_solver_returns_normal_equations_presum():
    solver = Solver.get_solver("normal_equations_presum")

    assert isinstance(solver, NormalEquationsPreSum)


def test_get_solver_with_args():
    solver = Solver.get_solver(
        "gain_substitution",
        refant=2,
        phase_only=True,
    )

    assert solver.refant == 2
    assert solver.phase_only is True


def test_get_solver_with_kwargs():
    solver = Solver.get_solver("gain_substitution", niter=100, tol=1e-8)

    assert solver.niter == 100
    assert solver.tol == 1e-8


def test_get_solver_raises_value_for_invalid_solver():
    with pytest.raises(ValueError):
        Solver.get_solver("invalid_solver")


def test_should_raise_not_implemented_exception():
    solver = Solver()
    with pytest.raises(NotImplementedError):
        solver.solve(
            "vis_vis",
            "vis_flags",
            "vis_weight",
            "model_vis",
            "model_flags",
            "gain_gain",
            "gain_weight",
            "gain_residual",
            "ant1",
            "ant2",
        )
