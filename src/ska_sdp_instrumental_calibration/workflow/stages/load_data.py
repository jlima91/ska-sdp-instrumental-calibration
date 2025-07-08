import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.workflow.utils import (
    create_bandpass_table,
)

from ...data_managers.dask_wrappers import load_ms

logger = logging.getLogger()


@ConfigurableStage(
    "load_data",
    configuration=Configuration(
        fchunk=ConfigParam(
            int,
            32,
            description="Number of frequency channels per chunk",
        ),
        ack=ConfigParam(
            bool,
            False,
            description="""Ask casacore to acknowledge each table operation""",
        ),
        start_chan=ConfigParam(
            int, 0, description="""Starting channel to read"""
        ),
        end_chan=ConfigParam(int, 0, description="""End channel to read"""),
        datacolumn=ConfigParam(
            str,
            "DATA",
            description="""MS data column to read DATA, CORRECTED_DATA, or
                    MODEL_DATA""",
            allowed_values=["DATA", "CORRECTED_DATA", "MODEL_DATA"],
        ),
        selected_sources=ConfigParam(
            list, None, description="""Sources to select"""
        ),
        selected_dds=ConfigParam(
            list, None, description="""Data descriptors to select"""
        ),
        average_channels=ConfigParam(
            bool, False, description="""Average all channels read"""
        ),
    ),
)
def load_data_stage(
    upstream_output,
    fchunk,
    ack,
    start_chan,
    end_chan,
    datacolumn,
    selected_sources,
    selected_dds,
    average_channels,
    _cli_args_,
):
    """
    Load the Measurement Set data.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        fchunk: int
            Number of frequency channels per chunk
        ack: bool
            Ask casacore to acknowledge each table operation
        start_chan: int
            Starting channel to read
        end_chan: int
            End channel to read4
        datacolumn: str
            MS data column to read DATA, CORRECTED_DATA, or
            MODEL_DATA
        selected_sources: list
            Sources to select
        selected_dds: list
            Data descriptors to select
        average_channels: bool
            Average all channels read"
        _cli_args_: dict
            CLI Arguments.

    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    input_ms = _cli_args_["input"]
    logger.info(f"Will read from {input_ms} in {fchunk}-channel chunks")

    vis = load_ms(
        input_ms,
        fchunk,
        start_chan=start_chan,
        ack=ack,
        datacolumn=datacolumn,
        end_chan=end_chan,
        selected_sources=selected_sources,
        selected_dds=selected_dds,
        average_channels=average_channels,
    )
    gaintable = create_bandpass_table(vis)
    upstream_output["vis"] = vis
    upstream_output["corrected_vis"] = vis
    upstream_output["gaintable"] = gaintable.chunk({"frequency": fchunk})
    upstream_output["beams"] = None
    return upstream_output
