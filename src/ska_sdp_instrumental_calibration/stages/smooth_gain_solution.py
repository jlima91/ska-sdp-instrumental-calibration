from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..plot import PlotGaintableFrequency
from ..xarray_processors.gain_smoothing import sliding_window_smooth
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import PlotSmoothGainsConfig


@ConfigurableStage(name="smooth_gain_solution", optional=True)
def smooth_gain_solution_stage(
    _upstream_output_,
    _output_dir_,
    plot_config: Annotated[
        PlotSmoothGainsConfig,
        Field(
            description="""Plot parameters""",
            default_factory=PlotSmoothGainsConfig,
        ),
    ],
    window_size: Annotated[
        int,
        Field(
            description="""Sliding window size.""",
        ),
    ] = 1,
    mode: Annotated[
        Literal["mean", "median"],
        Field(
            description="""Mode of smoothing""",
        ),
    ] = "median",
    export_gaintable: Annotated[
        bool,
        Field(
            description="""Export intermediate gain solutions.""",
        ),
    ] = False,
):
    """
    Smooth the gain solution.

    Parameters
    ----------
     _upstream_output_: dict
        Output from the upstream stage
    _output_dir_ : str
        Directory path where the output file will be written.
    plot_config: PlotConfig
        Configuration required for plotting.
    window_size: int
        Size of the window for running window smoothing
    mode: str
        Mode of smoothing. [mean or median]
    export_gaintable: bool
        Export intermediate gain solutions

    Returns
    -------
    dict
        Updated upstream_output with gaintable
    """
    _upstream_output_.add_checkpoint_key("gaintable")

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("smooth"):
        call_counter_suffix = f"_{call_count}"

    _upstream_output_.gaintable = sliding_window_smooth(
        _upstream_output_.gaintable, window_size, mode
    )

    if plot_config.plot_table:
        path_prefix = get_plots_path(
            _output_dir_,
            f"{plot_config.plot_path_prefix}{call_counter_suffix}",
        )
        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                _upstream_output_.gaintable,
                figure_title=plot_config.plot_title,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, f"smooth_gain{call_counter_suffix}.gaintable.h5parm"
        )
        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                _upstream_output_.gaintable, gaintable_file_path
            )
        )

    _upstream_output_.increment_call_count("smooth")

    return _upstream_output_
