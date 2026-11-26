import logging
from typing import Annotated, Literal

from pydantic import Field
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export.export_visibilities import (
    export_msv4_to_ps,
    msv4_to_visibility,
)
from ..scheduler import delayed
from ..xarray_processors.apply import apply_gaintable_to_dataset
from ._utils import get_visibilities_path

logger = logging.getLogger(__name__)


def _export_visibility_to_ms(msname, vis_list):
    """
    Export visibilities to MSv2 file, after adding the ``time_centroid``
    data variable required by ``export_visibility_to_ms``.

    ``time_centroid`` is dropped from the visibility templates, so it is
    set to the ``time`` broadcast to the vis dimensions, excluding
    polarisation.

    Parameters
    -----------
        msname: str
            Path of the MSv2 file to write.
        vis_list: list[Visibility]
            Visibilities to export.
    """
    vis_with_time_centroid = []
    for vis in vis_list:
        vis_per_pol = vis.vis.isel(polarisation=0, drop=True)
        time_centroid = vis.time.broadcast_like(vis_per_pol).transpose(
            *vis_per_pol.dims
        )
        # Assigned as (dims, data), so that xarray does not merge the
        # baselines MultiIndex coordinates of time_centroid again
        vis_with_time_centroid.append(
            vis.assign(time_centroid=(time_centroid.dims, time_centroid.data))
        )

    export_visibility_to_ms(msname, vis_with_time_centroid)


def _wait_for_ps_write(ps_path, _written):
    """
    Deferred task completing the Processing Set export.

    ``export_msv4_to_ps`` returns the data write as a lazy dask task. The
    task manager only computes dask collections that are arguments of a
    registered deferred task, so passing the write here registers it, in
    the same graph as the other tasks. It also makes the pipeline wait for
    it: with a distributed scheduler, the task manager persist does not
    block, and this task only runs once the write is complete.

    Parameters
    -----------
        ps_path: str
            Path of the Processing Set written.
        _written: dask.array.Array
            Task writing the Processing Set. Unused, see above.
    """
    logger.info("Exported visibilities to %s", ps_path)


def _export(vis, path_prefix, node_name, export_format):
    """
    Register the export of an MSv4 node in the requested formats.

    Parameters
    -----------
        vis: xr.DataTree
            MSv4 node to export.
        path_prefix: str
            Output path, without extension.
        node_name: str
            Name of the MSv4 node in the Processing Set.
        export_format: str
            ``ps``, ``ms`` or ``both``.
    """
    if export_format in ("ps", "both"):
        ps_path = f"{path_prefix}.ps.zarr"
        delayed(_wait_for_ps_write)(
            ps_path, export_msv4_to_ps(ps_path, vis, node_name)
        )

    if export_format in ("ms", "both"):
        delayed(_export_visibility_to_ms)(
            f"{path_prefix}.ms", [msv4_to_visibility(vis)]
        )


@ConfigurableStage(name="export_visibilities", optional=True)
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
    export_format: Annotated[
        Literal["ps", "ms", "both"],
        Field(
            description="""Format of the exported visibilities.
            Options are:
            1. ps: Processing Set (zarr), "<name>.ps.zarr"
            2. ms: MSv2, "<name>.ms"
            3. both: Processing Set and MSv2""",
        ),
    ] = "ps",
):
    """
    Export visibilities to a Processing Set and/or an MSv2.
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
        export_format: str
            Format of the exported visibilities (ps, ms, both).

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

    vis_prefix = "raw"
    if apply_gaintable_to_vis:
        vis_prefix = "corrected"
        gaintable = _upstream_output_["gaintable"]
        vis = apply_gaintable_to_dataset(vis, gaintable, inverse=True)
        _upstream_output_["corrected_vis"] = vis

    if data_to_export == "vis" or data_to_export == "all":
        name = f"{vis_prefix}{call_counter_suffix}"
        path_prefix = get_visibilities_path(_output_dir_, f"{prefix}/{name}")
        _export(vis, path_prefix, f"{prefix}_{name}", export_format)

    if data_to_export == "modelvis" or data_to_export == "all":
        modelvis = _upstream_output_["modelvis"]
        name = f"modelvis{call_counter_suffix}"
        path_prefix = get_visibilities_path(_output_dir_, f"{prefix}/{name}")
        _export(modelvis, path_prefix, f"{prefix}_{name}", export_format)

    _upstream_output_.increment_call_count("export_visibilities")

    return _upstream_output_
