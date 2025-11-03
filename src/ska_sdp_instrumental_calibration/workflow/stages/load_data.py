import logging
import os
from pathlib import Path

import dask
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    simplify_baselines_dim,
)
from ska_sdp_instrumental_calibration.data_managers.visibility import (
    check_if_cache_files_exist,
    read_dataset_from_zarr,
    write_dataset_to_zarr,
    write_ms_to_zarr,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    create_bandpass_table,
    get_vis_data,
    with_chunks,
)

logger = logging.getLogger(__name__)


@ConfigurableStage(
    "load_data",
    configuration=Configuration(
        nchannels_per_chunk=ConfigParam(
            int,
            32,
            nullable=False,
            description="""Number of frequency channels per chunk in the
            written zarr file. This is also the size of frequency chunk
            used across the pipeline.""",
        ),
        ntimes_per_ms_chunk=ConfigParam(
            int,
            5,
            nullable=False,
            description="""Number of time slots to include in each chunk
            while reading from measurement set.""",
        ),
        cache_directory=ConfigParam(
            str,
            None,
            nullable=True,
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
        fave_init=ConfigParam(
            int,
            4,
            nullable=False,
            description="Initial averaging on input. Will average the data only"
            "this value is greater than 1.",
        ),
        baselines_to_remove=ConfigParam(
            list,
            None,
            nullable=True,
            description="Baseline ids to remove from the input visibility",
        ),
    ),
)
def load_data_stage(
    upstream_output,
    nchannels_per_chunk,
    ntimes_per_ms_chunk,
    cache_directory,
    ack,
    datacolumn,
    field_id,
    data_desc_id,
    fave_init,
    baselines_to_remove,
    _cli_args_,
    _output_dir_,
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
    fave_init: int
        Initial averaging on input
    _cli_args_: dict
        Piper builtin. Contains all CLI Arguments.
    _output_dir_: str
        Piper builtin. Stores the output directory path.

    Returns
    -------
    dict
        Updated upstream_output with the loaded visibility data
    """
    upstream_output.add_checkpoint_key("gaintable")
    input_ms = _cli_args_["input"]

    input_ms = os.path.realpath(input_ms)

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
        "time": ntimes_per_ms_chunk,
        "frequency": nchannels_per_chunk,
    }

    # Pipeline only works on frequency chunks
    # Its expected that later stages follow same chunking pattern
    vis_chunks = {
        **non_chunked_dims,
        "time": -1,
        "frequency": nchannels_per_chunk,
    }
    upstream_output["chunks"] = vis_chunks

    if cache_directory is None:
        logger.info(
            "Setting cache_directory to output directory: %s", _output_dir_
        )
        cache_directory = _output_dir_

    vis_cache_directory = os.path.join(
        cache_directory,
        f"{os.path.basename(input_ms)}_fid{field_id}_ddid{data_desc_id}_fave{fave_init}",
    )
    os.makedirs(vis_cache_directory, mode=0o755, exist_ok=True)

    if check_if_cache_files_exist(vis_cache_directory):
        logger.info(
            "Reading cached visibilities from path %s", vis_cache_directory
        )
    else:
        logger.warning(
            f"Loading visibilities from path {input_ms} into memory"
        )
        vis = get_vis_data(
            dataset=Path(input_ms),
            fave_init=fave_init,
            baselines_to_remove=baselines_to_remove,
        )
        vis = simplify_baselines_dim(vis)

        logger.info(
            "Writing converted visibilities to cache dir: %s",
            vis_cache_directory,
        )
        with dask.config.set(scheduler="threads"):
            writer = write_dataset_to_zarr(
                vis_cache_directory, zarr_chunks, vis
            )
            dask.compute(writer)

    vis = read_dataset_from_zarr(vis_cache_directory, vis_chunks)

    gaintable = create_bandpass_table(vis)

    upstream_output["vis"] = vis
    upstream_output["gaintable"] = gaintable.pipe(with_chunks, vis_chunks)
    upstream_output["beams"] = None

    return upstream_output
