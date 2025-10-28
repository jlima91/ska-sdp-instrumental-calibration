import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.processing_tasks.delay import (
    apply_delay,
    calculate_delay,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    get_gaintables_path,
    get_plots_path,
    plot_gains,
    plot_gaintable,
    plot_station_delays,
)

from ...data_managers.data_export import (
    export_clock_to_h5parm,
    export_gaintable_to_h5parm,
)


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
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def delay_calibration_stage(
    upstream_output, oversample, plot_config, export_gaintable, _output_dir_
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
        export_gaintable: bool
            Export intermediate gain solutions
        _output_dir_ : str
            Directory path where the output file will be written.
    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """
    upstream_output.add_checkpoint_key("gaintable")
    vis = upstream_output["vis"]
    gaintable = upstream_output["gaintable"]

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("delay"):
        call_counter_suffix = f"_{call_count}"

    delaytable = calculate_delay(gaintable, oversample)

    gaintable = apply_delay(gaintable, delaytable)

    if plot_config["plot_table"]:
        path_prefix = get_plots_path(
            _output_dir_, f"delay{call_counter_suffix}"
        )

        upstream_output.add_compute_tasks(
            plot_gaintable(
                gaintable,
                path_prefix,
                figure_title="Delay",
                fixed_axis=plot_config["fixed_axis"],
            )
        )

        upstream_output.add_compute_tasks(
            plot_station_delays(
                delaytable,
                path_prefix,
            ),
            plot_gains(
                vis,
                gaintable,
                path_prefix,
            ),
            plot_station_delays(
                delaytable,
                path_prefix,
            ),
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, f"delay{call_counter_suffix}.gaintable.h5parm"
        )

        delaytable_file_path = get_gaintables_path(
            _output_dir_, f"delay{call_counter_suffix}.clock.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

        upstream_output.add_compute_tasks(
            export_clock_to_h5parm(delaytable, delaytable_file_path)
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("delay")

    return upstream_output
