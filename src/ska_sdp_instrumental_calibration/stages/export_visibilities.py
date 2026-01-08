import dask
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ..xarray_processors.apply import apply_gaintable_to_dataset
from ._utils import get_visibilities_path


@ConfigurableStage(
    "export_visibilities",
    configuration=Configuration(
        data_to_export=ConfigParam(
            str,
            default="all",
            description="""Select which visibilities to export.
            Options are:
            1. vis: Input calibrator visibilities
            2. modelvis: Model visibilities computed in INST pipeline
            3. all: both vis and modelvis
            """,
            allowed_values=["all", "vis", "modelvis"],
            nullable=False,
        ),
        apply_gaintable_to_vis=ConfigParam(
            bool,
            default=True,
            description="""Whether to apply gaintable (computed till
            this stage) to 'vis', before exporting""",
            nullable=False,
        ),
    ),
)
def export_visibilities_stage(
    upstream_output, data_to_export, apply_gaintable_to_vis, _output_dir_
):
    """
    Export visibilities to MSv2 file.
    The visibilities will be exported to a new subdirectory "visibilities"
    under the pipeline's output directory.
    Optionally one can apply the gaintable to raw visibilities before
    exporting.

    Parameters
    -----------
        upstream_output: dict
            Output from upstream stage.
        data_to_export: str
            Data to export (vis, modelvis, all).
        apply_gaintable_to_vis: bool
            Whether to apply gaintable to vis before exporting
        _output_dir_ : str
            Directory path where the output file will be written.

    Returns
    --------
        dict
            Upstream output with corrected vis and/or modelvis.
    """
    vis = upstream_output["vis"]

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("export_visibilities"):
        call_counter_suffix = f"_{call_count}"

    vis_prefix = "raw_"
    if apply_gaintable_to_vis:
        vis_prefix = "corrected_"
        gaintable = upstream_output["gaintable"]
        vis = apply_gaintable_to_dataset(vis, gaintable, inverse=True)
        upstream_output["corrected_vis"] = vis

    if data_to_export == "vis" or data_to_export == "all":
        path_prefix = get_visibilities_path(
            _output_dir_, f"{vis_prefix}vis{call_counter_suffix}.ms"
        )
        upstream_output.add_compute_tasks(
            dask.delayed(export_visibility_to_ms)(path_prefix, [vis])
        )

    if data_to_export == "modelvis" or data_to_export == "all":
        modelvis = upstream_output["modelvis"]
        path_prefix = get_visibilities_path(
            _output_dir_, f"modelvis{call_counter_suffix}.ms"
        )
        upstream_output.add_compute_tasks(
            dask.delayed(export_visibility_to_ms)(path_prefix, [modelvis])
        )

    upstream_output.increment_call_count("export_visibilities")

    return upstream_output
