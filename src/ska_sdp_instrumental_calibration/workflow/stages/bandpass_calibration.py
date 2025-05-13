import os
from copy import deepcopy

from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import plot_gaintable

from ...data_managers.dask_wrappers import run_solver
from ._common import RUN_SOLVER_DOCSTRING, RUN_SOLVER_NESTED_CONFIG


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

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("bandpass"):
        call_counter_suffix = f"_{call_count}"

    gaintable = run_solver(
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
        path_prefix = os.path.join(
            _output_dir_, f"bandpass{call_counter_suffix}"
        )
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
    upstream_output.increment_call_count("bandpass")
    return upstream_output


bandpass_calibration_stage.__doc__ = bandpass_calibration_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
