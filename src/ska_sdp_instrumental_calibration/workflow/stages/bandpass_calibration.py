import logging

import dask
from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import (
    get_gaintables_path,
    get_plots_path,
    parse_reference_antenna,
    plot_gains,
    plot_gaintable,
    plot_vis,
)

from ...data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    run_solver,
)
from ...data_managers.data_export import export_gaintable_to_h5parm
from ._common import BANDPASS_COMMON_CONFIG, RUN_SOLVER_DOCSTRING

logger = logging.getLogger()


def bandpass_calibration(
    upstream_output,
    run_solver_config,
    plot_config,
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
    upstream_output.add_checkpoint_key("gaintable")
    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

    refant = run_solver_config["refant"]
    run_solver_config["refant"] = parse_reference_antenna(refant, initialtable)

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
        calvis = apply_gaintable_to_dataset(vis, gaintable, inverse=True)
        path_prefix = get_plots_path(
            _output_dir_, f"bandpass{call_counter_suffix}"
        )

        upstream_output.add_compute_tasks(
            plot_gaintable(
                gaintable,
                path_prefix,
                figure_title="Bandpass",
                fixed_axis=plot_config["fixed_axis"],
                all_station_plot=True,
            ),
            plot_gains(
                vis,
                gaintable,
                path_prefix,
            ),
            plot_vis(
                vis,
                calvis,
                modelvis,
                path_prefix,
            ),
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
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


bandpass_calibration.__doc__ = bandpass_calibration.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)

bandpass_calibration_stage = ConfigurableStage(
    "bandpass_calibration",
    configuration=Configuration(**BANDPASS_COMMON_CONFIG),
)(bandpass_calibration)
