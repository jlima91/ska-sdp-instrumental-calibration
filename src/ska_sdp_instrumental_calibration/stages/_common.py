"""
Common configuration elements shared between different stages.
"""

from ska_sdp_piper.piper.configurations import ConfigParam

RUN_SOLVER_COMMON = dict(
    solver=ConfigParam(
        str,
        "gain_substitution",
        description="""Calibration algorithm to use.
                Options are:
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
                and solution intervals.""",
        allowed_values=[
            "gain_substitution",
            "jones_substitution",
            "normal_equations",
            "normal_equations_presum",
        ],
    ),
    refant=ConfigParam(
        (int, str),
        0,
        description="""Reference antenna.
                Currently only activated for gain_substitution solver""",
        nullable=False,
    ),
    niter=ConfigParam(
        int,
        50,
        description="""Number of solver iterations.""",
        nullable=False,
    ),
    phase_only=ConfigParam(
        bool,
        False,
        description="""Solve only for the phases. This can be set
                to ``True`` when solver is "gain_substitution",
                otherwise it must be ``False``.""",
        nullable=False,
    ),
    tol=ConfigParam(
        float,
        1e-06,
        description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance.""",
        nullable=False,
    ),
    crosspol=ConfigParam(
        bool,
        False,
        description="""Do solutions including cross polarisations
                i.e. XY, YX or RL, LR.
                Only used by "gain_substitution" solver.""",
        nullable=False,
    ),
    normalise_gains=ConfigParam(
        str,
        None,
        description="""Normalises the gains.
                Only available when solver is "gain_substitution".
                Possible types of normalization are: "mean", "median".
                To perform no normalization, set this to ``null``.
                """,
        allowed_values=[None, "mean", "median"],
        nullable=True,
    ),
    timeslice=ConfigParam(
        float,
        None,
        description="""Defines time scale over which each gain solution
                is valid. This is used to define time axis of the GainTable.
                This parameter is interpreted as follows,

                float: this is a custom time interval in seconds.
                Input timestamps are grouped by intervals of this duration,
                and said groups are separately averaged to produce
                the output time axis.

                ``None``: match the time resolution of the input, i.e. copy
                the time axis of the input Visibility""",
    ),
)

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
                normalise_gains: str
                    Normalises the gains.
                    options are None, "mean", "median".
                    None means no normalization.
                    Only available with gain_substitution.
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

PREDICT_VISIBILITIES_COMMON_CONFIG = dict(
    beam_type=ConfigParam(
        str,
        "everybeam",
        description="Type of beam model to use.",
    ),
    normalise_at_beam_centre=ConfigParam(
        bool,
        True,
        description="""If true, before running calibration, multiply vis
            and model vis by the inverse of the beam response in the
            beam pointing direction.""",
    ),
    eb_ms=ConfigParam(
        str,
        None,
        description="""If beam_type is "everybeam" but input ms does
            not have all of the metadata required by everybeam, this parameter
            is used to specify a separate dataset to use when setting up
            the beam models.""",
    ),
    gleamfile=ConfigParam(
        str,
        None,
        description="""Specifies the location of gleam catalogue
            file gleamegc.dat""",
    ),
    lsm_csv_path=ConfigParam(
        str,
        None,
        description="""Specifies the location of CSV file containing the
            sky model. The CSV file should be in OSKAR CSV format.""",
    ),
    element_response_model=ConfigParam(
        str,
        "oskar_dipole_cos",
        description="""Type of element response model.
            Required if beam_type is 'everybeam'.
            Refer documentation for more detials
            https://everybeam.readthedocs.io/en/latest/tree/python/utils.html
            """,
    ),
    fov=ConfigParam(
        float,
        5.0,
        description="""Specifies the width of the cone used when
            searching for compoents, in units of degrees.""",
    ),
    flux_limit=ConfigParam(
        float,
        1.0,
        description="""Specifies the flux density limit used when
            searching for compoents, in units of Jy.""",
    ),
    alpha0=ConfigParam(
        float,
        -0.78,
        description="""Nominal alpha value to use when fitted data
            are unspecified..""",
    ),
)
