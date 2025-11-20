from ska_sdp_func_python.calibration.solvers import solve_gaintable


class Solver:
    solver = None

    def __init__(self, niter=30, tol=1e-6, **__):
        self.niter = niter
        self.tol = tol

    def solve(self, vis, modelvis, gaintable):
        # Deep copy as solve_gaintable mutates gaintable
        gaintable = gaintable.copy(deep=True)
        # TODO: Check if this copy can be avoided
        vis = vis.copy(deep=True)
        modelvis = modelvis.copy(deep=True)

        return solve_gaintable(
            vis=vis,
            modelvis=modelvis,
            gain_table=gaintable,
            solver=self.solver,
            **{
                "phase_only": False,
                "normalise_gains": None,
                **self.__dict__,
            },
        )


class GainSubstitution(Solver):
    solver = "gain_substitution"

    def __init__(
        self,
        crosspol=False,
        normalise_gains="mean",
        phase_only=False,
        refant=0,
        **kwargs,
    ):
        super(GainSubstitution, self).__init__(**kwargs)
        self.refant = refant
        self.phase_only = phase_only
        self.crosspol = crosspol
        self.normalise_gains = normalise_gains

    def solve(self, vis, modelvis, gaintable):
        if self.refant < 0 or self.refant >= len(gaintable.antenna):
            raise ValueError(f"Invalid refant: {self.refant}")

        return super(GainSubstitution, self).solve(vis, modelvis, gaintable)


class JonesSubtitution(Solver):
    solver = "jones_substitution"


class NormalEquation(Solver):
    solver = "normal_equations"


class NormalEquationsPreSum(Solver):
    solver = "normal_equations_presum"


class SolverFactory:
    _solvers = {
        "gain_substitution": GainSubstitution,
        "jones_substitution": JonesSubtitution,
        "normal_equations": NormalEquation,
        "normal_equations_presum": NormalEquationsPreSum,
    }

    @classmethod
    def get_solver(cls, solver="gain_substitution", **kwargs):
        return cls._solvers[solver](**kwargs)
