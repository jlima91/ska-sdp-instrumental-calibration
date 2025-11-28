class SolverFactory(type):
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

    _solvers = {}

    def __new__(cls, name, bases, attrs):
        new_class = super(SolverFactory, cls).__new__(cls, name, bases, attrs)
        if "_SOLVER_NAME_" in attrs:
            cls._solvers[attrs["_SOLVER_NAME_"]] = new_class

        return new_class

    @classmethod
    def get_solver(cls, solver="gain_substitution", **kwargs):
        if solver not in cls._solvers:
            raise ValueError(
                f"{solver} not definebd."
                f" Supported solvers: {', '.join(cls._solvers)}"
            )
        return cls._solvers[solver](**kwargs)
