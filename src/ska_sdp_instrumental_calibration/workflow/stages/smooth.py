import os

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
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
        plot_table=ConfigParam(
            bool, False, description="Plot the Smoothed gaintable."
        ),
    ),
)
def smooth_gain_solution_stage(
    upstream_output, window_size, mode, plot_table, _output_dir_
):
    """
    Smooth the gain solution.

    Parameters:
    -----------
    upstream_output: dict
            Output from the upstream stage
    window_size: int
            Size of the window for running window smoothing.
    mode: str
            Mode of smoothing. [mean or median].
    plot_table: bool
            Plot the Smoothed gaintable.
    _output_dir_ : str
            Directory path where the output file will be written.
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

    if plot_table:
        path_prefix = os.path.join(_output_dir_, "smoothed-gaintable")
        upstream_output.add_compute_tasks(
            plot_gaintable(
                smooth_gain,
                path_prefix,
                figure_title="Smoothed Gain",
                drop_cross_pols=False,
            )
        )
    upstream_output.gaintable = upstream_output.gaintable.assign(
        {"gain": smooth_gain}
    )

    return upstream_output
