import logging
import os
from typing import Annotated, Literal, Optional

import dask
from pydantic import Field
from ska_sdp_piper.piper.command import CLIArgument
from ska_sdp_piper.piper.v2.stage import ConfigurableStage

from ...data_managers.gaintable import create_gaintable_from_visibility
from ...data_managers.visibility import (
    check_if_cache_files_exist,
    read_visibility_from_zarr,
    write_ms_to_zarr,
)

logger = logging.getLogger(__name__)


@ConfigurableStage(name="target_load_data")
def load_data_stage(
    _upstream_output_,
    _output_dir_,
    input: Annotated[list[str], CLIArgument],
    nchannels_per_chunk: Annotated[
        int,
        Field(
            description="""Number of frequency channels per chunk in the
            written zarr file.""",
        ),
    ] = 32,
    ntimes_per_ms_chunk: Annotated[
        int,
        Field(
            description="""Number of time slots to include in each chunk
            while reading from measurement set and writing in zarr file.
            This is also the size of time chunk used across the pipeline.""",
        ),
    ] = 5,
    cache_directory: Annotated[
        Optional[str],
        Field(
            description="""Cache directory containing previously stored
            visibility datasets as zarr files. The directory should contain
            a subdirectory with same name as the input target ms file name,
            which internally contains the zarr and pickle files.

            If None, the input ms will be converted to zarr file,
            and this zarr file will be stored in a new 'cache'
            subdirectory under the provided output directory.""",
        ),
    ] = None,
    timeslice: Annotated[
        float,
        Field(
            description="""Defines time scale over which each gain solution
            is valid. This is used to define time axis of the GainTable.

            float: this is a custom time interval in seconds.
            Input timestamps are grouped by intervals of this duration
            and separately averaged to produce the output time axis.""",
        ),
    ] = 3.0,
    ack: Annotated[
        bool,
        Field(
            description="""Ask casacore to acknowledge each table operation""",
        ),
    ] = False,
    datacolumn: Annotated[
        Literal["DATA", "CORRECTED_DATA", "MODEL_DATA"],
        Field(
            description="""MS data column to read visibility data from.""",
        ),
    ] = "DATA",
    field_id: Annotated[
        int,
        Field(
            description="""Field ID of the data in measurement set""",
        ),
    ] = 0,
    data_desc_id: Annotated[
        int,
        Field(
            description="""Data Description ID of the data in
            measurement set""",
        ),
    ] = 0,
):
    """
    This stage loads the target visibility data from either (in order of
    preference):

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
    input: CLIArgument
        Input measurementset.
    nchannels_per_chunk: int
        Number of frequency channels per chunk in the
        written zarr file.
    ntimes_per_ms_chunk: int
        Number of time dimension to include in each chunk
        while reading from measurement set and writing in zarr file.
        This value is used across the pipeline,
        i.e. for zarr file and for the visibility dataset.
    cache_directory: str
        Cache directory containing previously stored
        visibility datasets as zarr files. The directory should contain
        a subdirectory with same name as the input target ms file name, which
        internally contains the zarr and pickle files.
        If None, the input ms will be converted to zarr file,
        and this zarr file will be stored in a new 'cache'
        subdirectory under the provided output directory.
    timeslice : float
        Defines time scale over which each gain solution is valid.
        This is used to define time axis of the GainTable. This
        parameter is interpreted as follows,
        float: this is a custom time interval in seconds. Input
        timestamps are grouped by intervals of this duration,
        and said groups are separately averaged to produce the
        output time axis.
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
        Updated upstream_output with the loaded target visibility data
    """
    input_ms = os.path.realpath(input[0])

    # Common dimensions across zarr and loaded visibility dataset
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }

    vis_chunks = {
        **non_chunked_dims,
        "time": ntimes_per_ms_chunk,
        "frequency": nchannels_per_chunk,
    }

    _upstream_output_["chunks"] = vis_chunks

    if cache_directory is None:
        logger.info(
            "Setting cache_directory to output directory: %s", _output_dir_
        )
        cache_directory = _output_dir_

    vis_cache_directory = os.path.join(
        cache_directory,
        f"{os.path.basename(input_ms)}_fid{field_id}_ddid{data_desc_id}",
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
    gaintable = create_gaintable_from_visibility(vis, timeslice, "G")

    _upstream_output_["timeslice"] = timeslice
    _upstream_output_["vis"] = vis
    _upstream_output_["gaintable"] = gaintable
    _upstream_output_["central_beams"] = None
    return _upstream_output_
