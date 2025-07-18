import logging
import os
from copy import deepcopy

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import plot_gaintable

from ...data_managers.dask_wrappers import run_solver
from ...data_managers.data_export import export_gaintable_to_h5parm
from ._common import RUN_SOLVER_DOCSTRING, RUN_SOLVER_NESTED_CONFIG

logger = logging.getLogger()


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
        visibility_key=ConfigParam(
            str,
            "vis",
            description="Visibility data to be used for calibration.",
            allowed_values=["vis", "corrected_vis"],
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def bandpass_calibration_stage(
    upstream_output,
    run_solver_config,
    plot_config,
    flagging,
    visibility_key,
    export_gaintable,
    _output_dir_,
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
        visibility_key: str
            Visibility data to be used for calibration.
        export_gaintable: bool
            Export intermediate gain solutions
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

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("bandpass"):
        call_counter_suffix = f"_{call_count}"

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initialtable,
        **run_solver_config,
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

    if export_gaintable:
        gaintable_file_path = os.path.join(
            _output_dir_, f"bandpass{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("bandpass")
    return upstream_output


bandpass_calibration_stage.__doc__ = bandpass_calibration_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
