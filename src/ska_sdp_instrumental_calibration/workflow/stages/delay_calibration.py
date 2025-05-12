import os

import dask.delayed
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.processing_tasks.delay import apply_delay
from ska_sdp_instrumental_calibration.workflow.utils import plot_gaintable


@ConfigurableStage(
    "delay_calibration",
    configuration=Configuration(
        oversample=ConfigParam(int, 16, description="Oversample rate"),
        plot_config=NestedConfigParam(
            "Plot parameters",
            plot_table=ConfigParam(
                bool, False, description="Plot the generated gaintable"
            ),
            fixed_axis=ConfigParam(
                bool, False, description="Limit amplitude axis to [0-1]"
            ),
        ),
    ),
)
def delay_calibration_stage(
    upstream_output, oversample, plot_config, _output_dir_
):
    """
    Performs delay calibration

    Parameters
    __________
        upstream_output: dict
            Output from the upstream stage
        oversample: int
            Oversample rate
        plot_config: dict
            Configuration required for plotting.
            eg: {plot_table: False, fixed_axis: False}
        _output_dir_ : str
            Directory path where the output file will be written.
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    gaintable = upstream_output["gaintable"]

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("delay"):
        call_counter_suffix = f"_{call_count}"

    gaintable = dask.delayed(apply_delay)(gaintable, oversample)

    if plot_config["plot_table"]:
        path_prefix = os.path.join(_output_dir_, f"delay{call_counter_suffix}")
        upstream_output.add_compute_tasks(
            plot_gaintable(
                gaintable,
                path_prefix,
                figure_title="Delay",
                fixed_axis=plot_config["fixed_axis"],
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("delay")

    return upstream_output
