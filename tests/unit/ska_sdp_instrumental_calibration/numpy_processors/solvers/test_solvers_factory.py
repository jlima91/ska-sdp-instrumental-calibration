import pytest

from ska_sdp_instrumental_calibration.numpy_processors.solvers import (
    alternative_solvers,
    gain_substitution_solver,
    solvers_factory,
)

JonesSubtitution = alternative_solvers.JonesSubtitution
NormalEquation = alternative_solvers.NormalEquation
NormalEquationsPreSum = alternative_solvers.NormalEquationsPreSum
GainSubstitution = gain_substitution_solver.GainSubstitution
SolverFactory = solvers_factory.SolverFactory


def test_get_solver_returns_gain_substitution():
    solver = SolverFactory.get_solver("gain_substitution")

    assert isinstance(solver, GainSubstitution)


def test_get_solver_returns_jones_substitution():
    solver = SolverFactory.get_solver("jones_substitution")

    assert isinstance(solver, JonesSubtitution)


def test_get_solver_returns_normal_equations():
    solver = SolverFactory.get_solver("normal_equations")

    assert isinstance(solver, NormalEquation)


def test_get_solver_returns_normal_equations_presum():
    solver = SolverFactory.get_solver("normal_equations_presum")

    assert isinstance(solver, NormalEquationsPreSum)


def test_get_solver_with_args():
    solver = SolverFactory.get_solver(
        "gain_substitution",
        refant=2,
        phase_only=True,
    )

    assert solver.refant == 2
    assert solver.phase_only is True


def test_get_solver_with_kwargs():
    solver = SolverFactory.get_solver("gain_substitution", niter=100, tol=1e-8)

    assert solver.niter == 100
    assert solver.tol == 1e-8


def test_get_solver_raises_value_for_invalid_solver():
    with pytest.raises(ValueError):
        SolverFactory.get_solver("invalid_solver")
