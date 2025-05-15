import os

from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import (
    apply_weights_on_gain,
    plot_gaintable,
)

from ...processing_tasks.gain_smoothing import sliding_window_smooth


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
        apply_weights=ConfigParam(
            bool,
            False,
            description="Should apply weights on gaintable",
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
    upstream_output,
    window_size,
    mode,
    apply_weights,
    plot_config,
    _output_dir_,
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
    apply_weights: bool
            Should apply weights on gaintable.
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
    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("smooth"):
        call_counter_suffix = f"_{call_count}"

    if apply_weights:
        upstream_output.gaintable = apply_weights_on_gain(
            upstream_output.gaintable
        )

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
    upstream_output.increment_call_count("smooth")

    return upstream_output
