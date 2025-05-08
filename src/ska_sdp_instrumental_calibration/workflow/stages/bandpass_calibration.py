import os
from copy import deepcopy

import dask.delayed
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import plot_gaintable

from ...data_managers.dask_wrappers import run_solver

RUN_SOLVER_NESTED_CONFIG = NestedConfigParam(
    "Run Solver parameters",
    solver=ConfigParam(
        str,
        "gain_substitution",
        description="""Calibration algorithm to use.
                (default="gain_substitution")
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
        int,
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
    jones_type=ConfigParam(
        str,
        "T",
        description="""Type of calibration matrix T or G or B.
                The frequency axis of the output GainTable
                depends on the value provided:
                "B": the output frequency axis is the same as
                that of the input Visibility.
                "T" or "G": the solution is assumed to be
                frequency-independent, and the frequency axis of the
                output contains a single value: the average frequency
                of the input Visibility's channels.""",
        allowed_values=["T", "G", "B"],
        nullable=False,
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
            Configuration required for bandpass calibration.
                solver: str
                    Calibration algorithm to use (default="gain_substitution")
                    options are:
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
                    Reference antenna (default 0). Currently only activated for
                    the gain_substitution solver.
                niter: int
                    Maximum number of iterations (default=30)
                phase_only: bool
                    Solve only for the phases. default=True when
                    solver="gain_substitution", otherwise it must be False.
                tol: float
                    Iteration stops when the fractional change in the
                    gain solution is below this tolerance (default=1e-6).
                crosspol: bool
                    Do solutions including cross polarisations
                    i.e. XY, YX or RL, LR. Only used by the gain_substitution.
                normalise_gains: str
                    Normalises the gains (default="mean").
                    options are None, "mean", "median".
                    None means no normalization.
                    Only available with gain_substitution.
                jones_type: str
                    Type of calibration matrix T or G or B
                    The frequency axis of the output GainTable
                    depends on the value provided:
                    "B": the output frequency axis is the same as
                    that of the input Visibility.
                    "T" or "G": the solution is assumed to be
                    frequency-independent, and the frequency axis of the
                    output contains a single value: the average frequency
                    of the input Visibility's channels.
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
""".lstrip()


@ConfigurableStage(
    "bandpass_calibration",
    configuration=Configuration(
        run_solver_config=deepcopy(RUN_SOLVER_NESTED_CONFIG),
        plot_config=NestedConfigParam(
            "Plot parameters",
            plot_table=ConfigParam(
                bool,
                False,
                description="Plot the generated gaintable",
                nullable=False,
            ),
            fixed_axis=ConfigParam(
                bool,
                False,
                description="Limit amplitude axis to [0-1]",
                nullable=False,
            ),
        ),
        flagging=ConfigParam(
            bool, False, description="Run RFI flagging", nullable=False
        ),
    ),
)
def bandpass_calibration_stage(
    upstream_output, run_solver_config, plot_config, flagging, _output_dir_
):
    """
    Performs Bandpass Calibration

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        run_solver_config: dict
            {run_solver_docstring}
        plot_config: dict
            Configuration required for plotting.
            eg: {{plot_table: False, fixed_axis: False}}
        flagging: bool
            Run Flagging for time
        _output_dir_ : str
            Directory path where the output file will be written.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    # [TODO] if predict_vis stage is not run, obtain modelvis from data.
    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable
    vis = upstream_output.vis

    # [TODO] Remove this section once model_rotations returns xarray
    run_solver_func = (
        dask.delayed(run_solver)
        if hasattr(initialtable, "dask")
        else run_solver
    )

    gaintable = run_solver_func(
        vis=vis,
        modelvis=modelvis,
        gaintable=initialtable,
        solver=run_solver_config["solver"],
        niter=run_solver_config["niter"],
        refant=run_solver_config["refant"],
        phase_only=run_solver_config["phase_only"],
        tol=run_solver_config["tol"],
        crosspol=run_solver_config["crosspol"],
        normalise_gains=run_solver_config["normalise_gains"],
        jones_type=run_solver_config["jones_type"],
        timeslice=run_solver_config["timeslice"],
    )

    if plot_config["plot_table"]:
        path_prefix = os.path.join(_output_dir_, "bandpass")
        upstream_output.add_compute_tasks(
            plot_gaintable(
                gaintable,
                path_prefix,
                figure_title="Bandpass",
                fixed_axis=plot_config["fixed_axis"],
                all_station_plot=True,
            )
        )

    upstream_output["gaintable"] = gaintable

    return upstream_output


bandpass_calibration_stage.__doc__ = bandpass_calibration_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
