import os

from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import plot_gaintable


@ConfigurableStage(
    "smooth_gain_solution",
    configuration=Configuration(
        window_size=ConfigParam(int, None, description="Sliding window size."),
        mode=ConfigParam(
            str,
            "median",
            description="Mode of smoothing",
            allowed_values=["mean", "median"],
        ),
        plot_config=NestedConfigParam(
            "Plot parameters",
            plot_table=ConfigParam(
                bool, False, description="Plot the smoothed gaintable"
            ),
            plot_path_prefix=ConfigParam(
                str,
                "smoothed-gain",
                description="Path prefix to store smoothed gain plots",
            ),
            plot_title=ConfigParam(
                str,
                "Smoothed Gain",
                description="Title for smoothed gain plots",
            ),
        ),
    ),
)
def smooth_gain_solution_stage(
    upstream_output, window_size, mode, plot_config, _output_dir_
):
    """
    Smooth the gain solution.

    Parameters:
    -----------
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
    _output_dir_ : str
            Directory path where the output file will be written
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """
    rolled_gain = upstream_output.gaintable.gain.rolling(
        frequency=window_size, center=True
    )

    if mode == "mean":
        smooth_gain = rolled_gain.mean()
    else:
        smooth_gain = rolled_gain.median()

    upstream_output.gaintable = upstream_output.gaintable.assign(
        {"gain": smooth_gain}
    )

    if plot_config["plot_table"]:
        path_prefix = os.path.join(
            _output_dir_, plot_config["plot_path_prefix"]
        )
        upstream_output.add_compute_tasks(
            plot_gaintable(
                upstream_output.gaintable,
                path_prefix,
                figure_title=plot_config["plot_title"],
                drop_cross_pols=False,
            )
        )

    return upstream_output
