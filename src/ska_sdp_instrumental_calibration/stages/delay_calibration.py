from typing import Annotated

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import (
    export_clock_to_h5parm,
    export_gaintable_to_h5parm,
)
from ..plot import PlotGaintableFrequency, plot_station_delays
from ..xarray_processors.delay import apply_delay, calculate_delay
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import PlotConfig


@ConfigurableStage(name="delay_calibration")
def delay_calibration_stage(
    _upstream_output_,
    _qa_dir_,
    plot_config: Annotated[
        PlotConfig,
        Field(description="Plot parameters", default_factory=PlotConfig),
    ],
    oversample: Annotated[
        int,
        Field(description="Oversample rate"),
    ] = 1,
    export_gaintable: Annotated[
        bool,
        Field(description="Export intermediate gain solutions."),
    ] = True,
):
    """
    Performs delay calibration

    Parameters
    __________
         _upstream_output_: dict
            Output from the upstream stage
        _qa_dir_ : str
            Directory path where the diagnostic QA outputs will be written.
        plot_config: PlotConfig
            Configuration required for plotting.
            eg: {plot_table: False, fixed_axis: False}
        oversample: int
            Oversample rate
        export_gaintable: bool
            Export intermediate gain solutions
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """
    _upstream_output_.add_checkpoint_key("gaintable")

    gaintable = _upstream_output_["gaintable"]
    prefix = _upstream_output_.ms_prefix

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("delay"):
        call_counter_suffix = f"_{call_count}"

    delaytable = calculate_delay(gaintable, oversample)

    gaintable = apply_delay(gaintable, delaytable)

    if plot_config.plot_table:
        path_prefix = get_plots_path(
            _qa_dir_, f"{prefix}/delay{call_counter_suffix}"
        )

        freq_plotter = PlotGaintableFrequency(
            path_prefix=path_prefix,
        )

        _upstream_output_.add_compute_tasks(
            freq_plotter.plot(
                gaintable,
                figure_title="Delay",
                fixed_axis=plot_config.fixed_axis,
            )
        )

        _upstream_output_.add_compute_tasks(
            plot_station_delays(
                delaytable,
                path_prefix,
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _qa_dir_,
            f"{prefix}/delay{call_counter_suffix}.gaintable.h5parm",
        )

        delaytable_file_path = get_gaintables_path(
            _qa_dir_, f"{prefix}/delay{call_counter_suffix}.clock.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

        _upstream_output_.add_compute_tasks(
            export_clock_to_h5parm(delaytable, delaytable_file_path)
        )

    _upstream_output_["gaintable"] = gaintable
    _upstream_output_.increment_call_count("delay")

    return _upstream_output_
