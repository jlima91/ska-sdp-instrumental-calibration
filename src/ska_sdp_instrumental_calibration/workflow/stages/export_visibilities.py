import os

import dask
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
)


@ConfigurableStage(
    "export_visibilities",
    configuration=Configuration(
        data_to_export=ConfigParam(
            str,
            default=None,
            description="Visibilities to export",
            allowed_values=["all", "vis", "modelvis", None],
            nullable=True,
        ),
        apply_gaintable_to_vis=ConfigParam(
            bool,
            default=False,
            description="Apply gaintable to vis",
            nullable=True,
        ),
    ),
)
def export_visibilities_stage(
    upstream_output, data_to_export, apply_gaintable_to_vis, _output_dir_
):
    """
    Apply gaintable and export visibilities.

    Parameters
    -----------
        upstream_output: dict
            Output from upstream stage.
        data_to_export: str
            Data to export (vis, modelvis, all).
        apply_gaintable_to_vis: bool
            Apply gaintable to vis.
        _output_dir_ : str
            Directory path where the output file will be written.

    Returns
    --------
        dict
            Upstream output with corrected vis and/or modelvis.
    """
    gaintable = upstream_output["gaintable"]
    vis = upstream_output["vis"]

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("export_visibilities"):
        call_counter_suffix = f"_{call_count}"

    if apply_gaintable_to_vis:
        vis = apply_gaintable_to_dataset(vis, gaintable)
        upstream_output["corrected_vis"] = vis

    if data_to_export == "vis" or data_to_export == "all":
        path_prefix = os.path.join(
            _output_dir_, f"corrected_vis{call_counter_suffix}.ms"
        )
        upstream_output.add_compute_tasks(
            dask.delayed(export_visibility_to_ms)(path_prefix, [vis])
        )

    if data_to_export == "modelvis" or data_to_export == "all":
        path_prefix = os.path.join(
            _output_dir_, f"corrected_modelvis{call_counter_suffix}.ms"
        )
        upstream_output.add_compute_tasks(
            dask.delayed(export_visibility_to_ms)(
                path_prefix, [upstream_output["modelvis"]]
            )
        )

    upstream_output.increment_call_count("export_visibilities")

    return upstream_output
