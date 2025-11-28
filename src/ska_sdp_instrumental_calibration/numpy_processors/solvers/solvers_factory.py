from .alternative_solvers import (
    JonesSubtitution,
    NormalEquation,
    NormalEquationsPreSum,
)
from .gain_substitution_solver import GainSubstitution


class SolverFactory:
    """
    Factory class for creating solver instances for different calibration
    algorithms.

    This class provides a unified interface to instantiate solver classes
    such as GainSubstitution, JonesSubtitution, NormalEquation, and
    NormalEquationsPreSum based on a string identifier. It is useful for
    selecting and configuring calibration solvers dynamically at runtime.

    Usage:
        solver = SolverFactory.get_solver("gain_substitution", *args, **kwargs)

    Supported solvers:
        - "gain_substitution"
        - "jones_substitution"
        - "normal_equations"
        - "normal_equations_presum"
    """

    _solvers = {
        "gain_substitution": GainSubstitution,
        "jones_substitution": JonesSubtitution,
        "normal_equations": NormalEquation,
        "normal_equations_presum": NormalEquationsPreSum,
    }

    @classmethod
    def get_solver(cls, solver="gain_substitution", **kwargs):
        if solver not in cls._solvers:
            raise ValueError(
                f"{solver} not definebd."
                f" Supported solvers: {','.join(cls._solvers)}"
            )
        return cls._solvers[solver](**kwargs)
