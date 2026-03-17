"""
Common configuration elements shared between different stages.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RunSolverCommon(BaseModel):
    """
    A model describing the Runsolver config
    """

    solver: Annotated[
        Literal[
            "gain_substitution",
            "jones_substitution",
            "normal_equations",
            "normal_equations_presum",
        ],
        Field(description="Calibration algorithm to use."),
    ] = "gain_substitution"
    refant: Annotated[
        int | str,
        Field(description="Reference antenna."),
    ] = 0
    niter: Annotated[
        int,
        Field(description="Number of solver iterations."),
    ] = 50
    phase_only: Annotated[
        bool,
        Field(description="Solve only for phases."),
    ] = False
    tol: Annotated[
        float,
        Field(description="Tolerance for solver convergence."),
    ] = 1e-6
    crosspol: Annotated[
        bool,
        Field(description="Include cross polarisations."),
    ] = False


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
                timeslice: float
                    Defines the time scale over which each gain solution is
                    valid. This is used to define time axis of the GainTable.
                    This parameter is interpreted as follows,
                    float: this is a custom time interval in seconds.
                    Input timestamps are grouped by intervals of this duration,
                    and said groups are separately averaged to produce
                    the output time axis.
                    None: match the time resolution of the input, i.e. copy
                    the time axis of the input Visibility
""".strip()


class PlotConfig(BaseModel):
    plot_table: Annotated[
        bool,
        Field(description="Plot the generated gaintable"),
    ] = True
    fixed_axis: Annotated[
        bool,
        Field(description="Limit amplitude axis to [0-1]"),
    ] = False
