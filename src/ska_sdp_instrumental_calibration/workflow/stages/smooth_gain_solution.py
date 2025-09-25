import os

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import plot_gaintable

from ...data_managers.data_export import export_gaintable_to_h5parm
from ...processing_tasks.gain_smoothing import sliding_window_smooth


@ConfigurableStage(
    "smooth_gain_solution",
    configuration=Configuration(
        window_size=ConfigParam(
            int, 1, description="Sliding window size.", nullable=False
        ),
        mode=ConfigParam(
            str,
            "median",
            description="Mode of smoothing",
            allowed_values=["mean", "median"],
            nullable=False,
        ),
        plot_config=NestedConfigParam(
            "Plot parameters",
            plot_table=ConfigParam(
                bool,
                False,
                description="Plot the smoothed gaintable",
                nullable=False,
            ),
            plot_path_prefix=ConfigParam(
                str,
                "smoothed-gain",
                description="Path prefix to store smoothed gain plots",
                nullable=False,
            ),
            plot_title=ConfigParam(
                str,
                "Smoothed Gain",
                description="Title for smoothed gain plots",
                nullable=False,
            ),
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def smooth_gain_solution_stage(
    upstream_output,
    window_size,
    mode,
    plot_config,
    export_gaintable,
    _output_dir_,
):
    """
    Smooth the gain solution.

    Parameters
    ----------
    upstream_output: dict
        Output from the upstream stage
    window_size: int
        Size of the window for running window smoothing
    mode: str
        Mode of smoothing. [mean or median]
    plot_config: dict
        Configuration required for plotting.
        {plot_table: False, plot_path_prefix: "smoothed-gain",
        plot_title: "Smooth gain"}
    export_gaintable: bool
        Export intermediate gain solutions
    _output_dir_ : str
        Directory path where the output file will be written

    Returns
    -------
    dict
        Updated upstream_output with gaintable
    """
    upstream_output.add_checkpoint_key("gaintable")

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("smooth"):
        call_counter_suffix = f"_{call_count}"

    upstream_output.gaintable = sliding_window_smooth(
        upstream_output.gaintable, window_size, mode
    )

    if plot_config["plot_table"]:
        path_prefix = os.path.join(
            _output_dir_,
            f"{plot_config['plot_path_prefix']}{call_counter_suffix}",
        )
        upstream_output.add_compute_tasks(
            plot_gaintable(
                upstream_output.gaintable,
                path_prefix,
                figure_title=plot_config["plot_title"],
                drop_cross_pols=False,
            )
        )

    if export_gaintable:
        gaintable_file_path = os.path.join(
            _output_dir_, f"smooth_gain{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                upstream_output.gaintable, gaintable_file_path
            )
        )

    upstream_output.increment_call_count("smooth")

    return upstream_output
