"""
Common configuration elements shared between different stages.
"""

RUN_SOLVER_DOCSTRING = """
            Configuration required for solver.
                solver: str
                    Calibration algorithm to use. Options are:
                    "gain_substitution" - original substitution algorithm
                    with separate solutions for each polarisation term.
                    "jones_substitution" - solve antenna-based Jones matrices
                    as a whole, with independent updates within each iteration.
                    "normal_equations" - solve normal equations within
                    each iteration formed from linearisation with respect to
                    antenna-based gain and leakage terms.
                    "normal_equations_presum" - same as normal_equations
                    option but with an initial accumulation of visibility
                    products over time and frequency for each solution
                    interval. This can be much faster for large datasets
                    and solution intervals.
                refant: int
                    Reference antenna. Currently only activated for
                    the gain_substitution solver.
                niter: int
                    Maximum number of iterations
                phase_only: bool
                    Solve only for the phases when solver="gain_substitution",
                    otherwise it must be False.
                tol: float
                    Iteration stops when the fractional change in the
                    gain solution is below this tolerance.
                crosspol: bool
                    Do solutions including cross polarisations
                    i.e. XY, YX or RL, LR. Only used by the gain_substitution.
""".strip()
