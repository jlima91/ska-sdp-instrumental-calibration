import logging
import os
from typing import Annotated, Literal, Optional

import dask
from pydantic import Field
from ska_sdp_piper.piper import CLIArgument, ConfigurableStage

from ska_sdp_instrumental_calibration.sdm import SDM

from ..data_managers.gaintable import create_gaintable_from_visibility
from ..data_managers.visibility import (
    check_if_cache_files_exist,
    read_ms_field_id,
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
    input: Annotated[list[str], CLIArgument],
    nchannels_per_chunk: Annotated[
        int,
        Field(
            description="""Number of frequency channels per chunk in the
            written zarr file. This is also the size of frequency chunk
            used across the pipeline."""
        ),
    ] = 32,
    ntimes_per_ms_chunk: Annotated[
        int,
        Field(
            description="""Number of time dimension to include in each chunk
            while reading from measurement set. This also sets
            the number of times per chunk for zarr file."""
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
            and this zarr file will be stored in a new 'cache'
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
       zarr file with chunks along frequency and use it as input to the
       pipeline. This zarr dataset will be stored in 'cache_directory' for
       later use.

    Parameters
    ----------
    _upstream_output_: dict
        Output from the upstream stage
    _output_dir_: str
        Piper builtin. Stores the output directory path.
    input: CLIArgument
        Input measurementset.
    nchannels_per_chunk: int
        Number of frequency channels per chunk in the
        written zarr file. This value is used across the pipeline,
        i.e. for zarr file and for the visibility dataset.
    ntimes_per_ms_chunk: int
        Number of time dimension to include in each chunk
        while reading from measurement set. This also sets
        the number of times per chunk for zarr file.
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
        for ms in input
    ]


def _load_data(
    _output_dir_,
    input_ms: str,
    nchannels_per_chunk: int,
    ntimes_per_ms_chunk: int,
    cache_directory: Optional[str],
    ack: bool,
    datacolumn: Literal["DATA", "CORRECTED_DATA", "MODEL_DATA"],
    field_id: int,
    data_desc_id: int,
) -> UpstreamOutput:

    _upstream_output_ = UpstreamOutput()
    _upstream_output_.add_checkpoint_key("gaintable")
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }

    # Its expected that later stages follow same chunking pattern
    vis_chunks = {
        **non_chunked_dims,
        "time": ntimes_per_ms_chunk,
        "frequency": nchannels_per_chunk,
    }
    _upstream_output_["chunks"] = vis_chunks
    ms_file = os.path.basename(input_ms)

    if cache_directory is None:
        logger.info(
            "Setting cache_directory to output directory: %s", _output_dir_
        )
        cache_directory = _output_dir_

    vis_cache_directory = os.path.join(
        cache_directory,
        f"{ms_file}_fid{field_id}_ddid{data_desc_id}",
    )
    os.makedirs(vis_cache_directory, mode=0o755, exist_ok=True)

    if check_if_cache_files_exist(vis_cache_directory):
        logger.info(
            "Reading cached visibilities from path %s", vis_cache_directory
        )
    else:
        logger.info(
            "Writing converted visibilities to cache dir: %s",
            vis_cache_directory,
        )
        with dask.annotate(resources={"process": 1}):
            write_ms_to_zarr(
                input_ms,
                vis_cache_directory,
                vis_chunks,
                ack=ack,
                datacolumn=datacolumn,
                field_id=field_id,
                data_desc_id=data_desc_id,
            )

    vis = read_visibility_from_zarr(vis_cache_directory, vis_chunks)

    _upstream_output_["ms_prefix"] = os.path.splitext(ms_file)[0]
    _upstream_output_["vis"] = vis
    _upstream_output_["gaintable"] = create_gaintable_from_visibility(
        vis, "full", "B"
    )
    _upstream_output_["central_beams"] = None
    _upstream_output_["beams_factory"] = None
    _upstream_output_["field_id"] = read_ms_field_id(input_ms)
    _upstream_output_["calibration_purpose"] = SDM.BANDPASS.value

    return _upstream_output_
