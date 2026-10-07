"""
Common configuration elements shared between different stages.
"""

from typing import Annotated, Literal

from pydantic import Field
from ska_sdp_piper.piper import PiperBaseModel


class RunSolverConfig(PiperBaseModel):
    """
    A model describing the Runsolver config
    """

    solver: Annotated[
        Literal[
            "gain_substitution",
            "jones_substitution",
            "normal_equations",
            "normal_equations_presum",
            "dp3_gaincal",
        ],
        Field(
            description="""Calibration algorithm to use. Options are:
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
                "dp3_gaincal" - DP3 gaincal solver. The DP3 calibration
                type is derived from crosspol and phase_only."""
        ),
    ] = "jones_substitution"
    refant: Annotated[
        int | str,
        Field(
            description="""Reference antenna.
              Currently only activated for gain_substitution solver"""
        ),
    ] = 0
    niter: Annotated[
        int,
        Field(description="Number of solver iterations."),
    ] = 50
    phase_only: Annotated[
        bool,
        Field(
            description="""Solve only for the phases. This can be set
                to ``True`` when solver is "gain_substitution" or
                "dp3_gaincal", otherwise it must be ``False``.
                For "dp3_gaincal", it can not be combined with
                crosspol."""
        ),
    ] = False
    tol: Annotated[
        float,
        Field(
            description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance."""
        ),
    ] = 1e-3
    crosspol: Annotated[
        bool,
        Field(
            description="""Do solutions including cross polarisations
                i.e. XY, YX or RL, LR.
                Only used by "gain_substitution" and "dp3_gaincal"
                solvers."""
        ),
    ] = False


class TargetRunSolverConfig(PiperBaseModel):
    """
    A model describing the Runsolver Configuration passed
    to the Complex Gain Calibration stage
    """

    refant: Annotated[
        int | str,
        Field(
            description="""Reference antenna.
                Currently only activated for gain_substitution solver"""
        ),
    ] = 0
    niter: Annotated[
        int,
        Field(description="Number of solver iterations."),
    ] = 200
    tol: Annotated[
        float,
        Field(
            description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance."""
        ),
    ] = 1e-6
    crosspol: Annotated[
        bool,
        Field(
            description="""Do solutions including cross polarisations
                i.e. XY, YX or RL, LR.
                Only used by "gain_substitution" solver."""
        ),
    ] = False
