import logging
import math
import os
import time
from typing import Annotated, Literal, Optional, Union

import dask
from pydantic import Field
from ska_sdp_piper.piper import CLIArgument, ConfigurableStage

from ..data_managers.chunking import (
    estimate_channels_per_chunk,
    get_cluster_spec,
)
from ..data_managers.gaintable import create_gaintable_from_visibility
from ..data_managers.visibility import (
    check_if_cache_files_exist,
    create_template_vis_from_ms,
    read_ms_field_id,
    read_vis_shape_from_zarr,
    read_visibility_from_zarr,
    write_ms_to_zarr,
)
from ..scheduler import UpstreamOutput
from ..tagger import Tags

logger = logging.getLogger(__name__)


@ConfigurableStage(name="load_data")
@Tags.BROADCASTER
def load_data_stage(
    _upstream_output_,
    _output_dir_,
    input_ms: Annotated[list[str], CLIArgument],
    nchannels_per_chunk: Annotated[
        Union[int, Literal["auto"]],
        Field(
            description="""Number of frequency channels per chunk in the
            written zarr file. This is also the size of frequency chunk
            used across the pipeline. If "auto", it is estimated from the
            visibility size and the threads and memory of the dask
            cluster."""
        ),
    ] = "auto",
    ntimes_per_ms_chunk: Annotated[
        int,
        Field(
            description="""Number of visibility time slots to include in each
            time chunk of zarr file."""
        ),
    ] = 5,
    cache_directory: Annotated[
        Optional[str],
        Field(
            description="""Cache directory containing previously stored
            visibility datasets as zarr files. The directory should contain
            a subdirectory with same name as the input ms file name, which
            internally contains the zarr and pickle files.
            If None, the input ms will be converted to zarr file,
            and this zarr file will be stored in a new '.cache'
            subdirectory under the provided output directory."""
        ),
    ] = None,
    ack: Annotated[
        bool,
        Field(description="Ask casacore to acknowledge each table operation."),
    ] = False,
    datacolumn: Annotated[
        Literal["DATA", "CORRECTED_DATA", "MODEL_DATA"],
        Field(
            description="Measurement set data column name to read data from."
        ),
    ] = "DATA",
    field_id: Annotated[
        int,
        Field(description="Field ID of the data in measurement set"),
    ] = 0,
    data_desc_id: Annotated[
        int,
        Field(
            description="Data Description ID of the data in measurement set"
        ),
    ] = 0,
) -> list[UpstreamOutput]:
    """
    This stage loads the visibility data from either (in order of preference):

    1. An existing dataset stored as a zarr file inside the 'cache_directory'.
    2. From input MSv2 measurement set. Here it will create an intemediate
       zarr file with chunks along frequency and time, then use it as input
       to the pipeline. This zarr dataset will be stored in 'cache_directory'
       for later use.

    Parameters
    ----------
    _upstream_output_: dict
        Output from the upstream stage
    _output_dir_: str
        Piper builtin. Stores the output directory path.
    input_ms: CLIArgument
        Input measurementset.
    nchannels_per_chunk: int | "auto"
        Number of frequency channels per chunk in the
        written zarr file. This value is used across the pipeline,
        i.e. for zarr file and for the visibility dataset.
        If "auto", it is estimated from the visibility size and the
        threads and memory of the dask cluster.
    ntimes_per_ms_chunk: int
        Number of visibility time slots to include in each
        time chunk of zarr file
    cache_directory: str
        Cache directory containing previously stored
        visibility datasets as zarr files. The directory should contain
        a subdirectory with same name as the input ms file name, which
        internally contains the zarr and pickle files.
        If None, the input ms will be converted to zarr file,
        and this zarr file will be stored in a new 'cache'
        subdirectory under the provided output directory.
    ack: bool
        Ask casacore to acknowledge each table operation
    datacolumn: str
        Measurement set data column name to read data from.
    field_id: int
        Field ID of the data in measurement set
    data_desc_id: int
        Data Description ID of the data in measurement set

    Returns
    -------
    dict
        Updated upstream_output with the loaded visibility data
    """
    return [
        _load_data(
            _output_dir_,
            os.path.realpath(ms),
            nchannels_per_chunk,
            ntimes_per_ms_chunk,
            cache_directory,
            ack,
            datacolumn,
            field_id,
            data_desc_id,
        )
        for ms in input_ms
    ]


def _load_data(
    _output_dir_,
    input_ms: str,
    nchannels_per_chunk: Union[int, Literal["auto"]],
    ntimes_per_ms_chunk: int,
    cache_directory: Optional[str],
    ack: bool,
    datacolumn: Literal["DATA", "CORRECTED_DATA", "MODEL_DATA"],
    field_id: int,
    data_desc_id: int,
) -> UpstreamOutput:

    _upstream_output_ = UpstreamOutput()
    ms_file = os.path.basename(input_ms)

    if cache_directory is None:
        cache_directory = os.path.join(_output_dir_, ".cache")
        logger.info("Setting cache_directory as %s", cache_directory)

    vis_cache_directory = os.path.join(
        cache_directory,
        f"{ms_file}_fid{field_id}_ddid{data_desc_id}",
    )
    os.makedirs(vis_cache_directory, mode=0o755, exist_ok=True)
    cache_exists = check_if_cache_files_exist(vis_cache_directory)

    if nchannels_per_chunk == "auto":
        if cache_exists:
            vis_shape = read_vis_shape_from_zarr(vis_cache_directory)
        else:
            vis_shape = create_template_vis_from_ms(
                input_ms,
                ack=ack,
                datacolumn=datacolumn,
                field_ids=[field_id],
                data_desc_ids=[data_desc_id],
            )[0].vis.shape
        nchannels_per_chunk = _estimate_nchannels_per_chunk(vis_shape)

    non_chunked_dims = {
        dim: -1
        for dim in [
            "time",
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }
    vis_chunks = {
        **non_chunked_dims,
        "frequency": nchannels_per_chunk,
    }
    zarr_chunks = {
        **vis_chunks,
        "time": ntimes_per_ms_chunk,
    }
    # Its expected that later stages follow same chunking pattern
    _upstream_output_["chunks"] = vis_chunks

    if cache_exists:
        logger.info(
            "Reading cached visibilities from path %s", vis_cache_directory
        )
    else:
        logger.info(
            "Writing converted visibilities to cache dir: %s",
            vis_cache_directory,
        )
        writer = write_ms_to_zarr(
            input_ms,
            vis_cache_directory,
            zarr_chunks,
            ack=ack,
            datacolumn=datacolumn,
            field_id=field_id,
            data_desc_id=data_desc_id,
        )
        logger.info("MSv2->Zarr Visibility cache conversion: starting")
        start_time = time.perf_counter()
        dask.compute(writer)
        logger.info(
            "MSv2->Zarr Visibility cache conversion: finished. "
            "duration_sec = %.3f",
            time.perf_counter() - start_time,
        )

    vis = read_visibility_from_zarr(vis_cache_directory, vis_chunks)

    logger.info("Visibility dataset dimensions: %s", dict(vis.sizes))

    _upstream_output_["ms_prefix"] = os.path.splitext(ms_file)[0]
    _upstream_output_["vis"] = vis
    _upstream_output_["gaintable"] = create_gaintable_from_visibility(
        vis, "full", "B"
    )
    _upstream_output_["central_beams"] = None
    _upstream_output_["beams_factory"] = None
    _upstream_output_["field_id"] = read_ms_field_id(input_ms)
    _upstream_output_["calibration_purpose"] = "bandpass"

    return _upstream_output_


def _estimate_nchannels_per_chunk(vis_shape: tuple[int, ...]) -> int:
    """
    Estimate frequency channels per chunk for visibilities of shape
    (time, baselineid, frequency, polarisation) on the current dask cluster.
    """
    ntimes, nbaselines, nchannels, npols = vis_shape
    cluster_spec = get_cluster_spec()

    nchannels_per_chunk = estimate_channels_per_chunk(
        nchannels, ntimes * nbaselines * npols, cluster_spec
    )
    logger.info(
        "Auto chunking: %d channels per chunk (%d chunks) for %d channels "
        "on %d threads with %.2f MiB memory per thread",
        nchannels_per_chunk,
        math.ceil(nchannels / nchannels_per_chunk),
        nchannels,
        cluster_spec.total_threads,
        cluster_spec.memory_per_thread_bytes / 2**20,
    )

    return nchannels_per_chunk
