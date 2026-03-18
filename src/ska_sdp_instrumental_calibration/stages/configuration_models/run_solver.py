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
        ],
        Field(description="Calibration algorithm to use."),
    ] = "jones_substitution"
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
    ] = 1e-3
    crosspol: Annotated[
        bool,
        Field(description="Include cross polarisations."),
    ] = False


class TargetRunSolverConfig(PiperBaseModel):
    """
    A model describing the Runsolver Configuration passed
    to the Complex Gain Calibration stage
    """

    refant: Annotated[
        int | str,
        Field(description="Reference antenna."),
    ] = 0
    niter: Annotated[
        int,
        Field(description="Number of solver iterations."),
    ] = 50
    tol: Annotated[
        float,
        Field(description="Tolerance for solver convergence."),
    ] = 1e-6
    crosspol: Annotated[
        bool,
        Field(description="Include cross polarisations."),
    ] = False
