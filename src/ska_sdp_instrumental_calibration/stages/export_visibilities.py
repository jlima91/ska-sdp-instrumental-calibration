from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms
from ska_sdp_piper.piper import ConfigurableStage

from ..xarray_processors.apply import apply_gaintable_to_dataset
from ._utils import get_visibilities_path


@ConfigurableStage(name="export_visibilities")
def export_visibilities_stage(
    _upstream_output_,
    _output_dir_,
    data_to_export: Annotated[
        Literal["all", "vis", "modelvis"],
        Field(
            description="""Select which visibilities to export.
            Options are:
            1. vis: Input calibrator visibilities
            2. modelvis: Model visibilities computed in INST pipeline
            3. all: both vis and modelvis""",
        ),
    ] = "all",
    apply_gaintable_to_vis: Annotated[
        bool,
        Field(
            description="""Whether to apply gaintable (computed till
            this stage) to 'vis', before exporting""",
        ),
    ] = True,
):
    """
    Export visibilities to MSv2 file.
    The visibilities will be exported to a new subdirectory "visibilities"
    under the pipeline's output directory.
    Optionally one can apply the gaintable to raw visibilities before
    exporting.

    Parameters
    -----------
         _upstream_output_: dict
            Output from the upstream stage
        _output_dir_ : str
            Directory path where the output file will be written.
        data_to_export: str
            Data to export (vis, modelvis, all).
        apply_gaintable_to_vis: bool
            Whether to apply gaintable to vis before exporting

    Returns
    --------
        dict
            Upstream output with corrected vis and/or modelvis.
    """
    vis = _upstream_output_["vis"]
    prefix = _upstream_output_.ms_prefix

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("export_visibilities"):
        call_counter_suffix = f"_{call_count}"

    vis_prefix = "raw_"
    if apply_gaintable_to_vis:
        vis_prefix = "corrected_"
        gaintable = _upstream_output_["gaintable"]
        vis = apply_gaintable_to_dataset(vis, gaintable, inverse=True)
        _upstream_output_["corrected_vis"] = vis

    if data_to_export == "vis" or data_to_export == "all":
        path_prefix = get_visibilities_path(
            _output_dir_, f"{vis_prefix}{prefix}{call_counter_suffix}.ms"
        )
        _upstream_output_.add_compute_tasks(
            dask.delayed(export_visibility_to_ms)(path_prefix, [vis])
        )

    if data_to_export == "modelvis" or data_to_export == "all":
        modelvis = _upstream_output_["modelvis"]
        path_prefix = get_visibilities_path(
            _output_dir_, f"{prefix}_modelvis{call_counter_suffix}.ms"
        )
        _upstream_output_.add_compute_tasks(
            dask.delayed(export_visibility_to_ms)(path_prefix, [modelvis])
        )

    _upstream_output_.increment_call_count("export_visibilities")

    return _upstream_output_
