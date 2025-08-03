import logging
import os

import dask
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.visibility import (
    check_if_cache_files_exist,
    read_dataset_from_zarr,
    write_ms_to_zarr,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    create_bandpass_table,
    with_chunks,
)

logger = logging.getLogger(__name__)


@ConfigurableStage(
    "load_data",
    configuration=Configuration(
        fchunk=ConfigParam(
            int,
            32,
            nullable=False,
            description="""Number of frequency channels per chunk in the
            written zarr file. This is also the size of frequency chunk
            used across the pipeline.""",
        ),
        times_per_ms_chunk=ConfigParam(
            int,
            5,
            nullable=False,
            description="""Number of time slots to include in each chunk
            while reading from measurement set.""",
        ),
        cache_directory=ConfigParam(
            str,
            None,
            nullable=False,
            description="""Cache directory containing previously stored
            visibility datasets as zarr files. The directory should contain
            a subdirectory with same name as the input ms file name, which
            internally contains the zarr and pickle files.
            If None, the input ms will be converted to zarr file,
            and this zarr file will be stored in a new 'cache'
            subdirectory under the provided output directory.""",
        ),
        ack=ConfigParam(
            bool,
            False,
            nullable=False,
            description="Ask casacore to acknowledge each table operation",
        ),
        datacolumn=ConfigParam(
            str,
            "DATA",
            nullable=False,
            description="MS data column to read visibility data from.",
            allowed_values=["DATA", "CORRECTED_DATA", "MODEL_DATA"],
        ),
        field_id=ConfigParam(
            int,
            0,
            nullable=False,
            description="Field ID of the data in measurement set",
        ),
        data_desc_id=ConfigParam(
            int,
            0,
            nullable=False,
            description="Data Description ID of the data in measurement set",
        ),
    ),
)
def load_data_stage(
    upstream_output,
    fchunk,
    times_per_ms_chunk,
    cache_directory,
    ack,
    datacolumn,
    field_id,
    data_desc_id,
    _cli_args_,
):
    """
    This stage loads the visibility data from either (in order of preference):

    1. An existing dataset stored as a zarr file inside the 'cache_directory'.
    2. From input MSv2 measurement set. Here it will create an intemediate
       zarr file with chunks along frequency and use it as input to the
       pipeline. This zarr dataset will be stored in 'cache_directory' for
       later use.

    Parameters
    ----------
    upstream_output: dict
        Output from the upstream stage
    fchunk: int
        Number of frequency channels per chunk in the
        written zarr file. This is also the fchunk value used across
        the pipeline.
    times_per_ms_chunk: int
        Number of time dimension to include in each chunk
        while reading from measurement set.
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
    _cli_args_: dict
        CLI Arguments.

    Returns
    -------
    dict
        Updated upstream_output with the loaded visibility data
    """
    input_ms = _cli_args_["input"]

    # Common dimensions across zarr and loaded visibility dataset
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }

    # This is chunking of the intermidiate zarr file
    zarr_chunks = {
        **non_chunked_dims,
        "time": times_per_ms_chunk,
        "frequency": fchunk,
    }

    # Pipeline only works on frequency chunks
    # Its expected that later stages follow same chunking pattern
    vis_chunks = {
        **non_chunked_dims,
        "time": -1,
        "frequency": fchunk,
    }
    upstream_output["chunks"] = vis_chunks

    if cache_directory is None:
        raise ValueError("Cache directory must be provided.")

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
                zarr_chunks,
                ack=ack,
                datacolumn=datacolumn,
                field_id=field_id,
                data_desc_id=data_desc_id,
            )

    vis = read_dataset_from_zarr(vis_cache_directory, vis_chunks)

    gaintable = create_bandpass_table(vis)
    upstream_output["vis"] = vis
    upstream_output["gaintable"] = gaintable.pipe(with_chunks, vis_chunks)
    upstream_output["beams"] = None
    return upstream_output
