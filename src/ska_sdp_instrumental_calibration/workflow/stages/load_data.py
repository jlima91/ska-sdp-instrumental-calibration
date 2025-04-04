import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    load_ms,
)

logger = logging.getLogger()


@ConfigurableStage(
    "load_data",
    configuration=Configuration(
        fchunk=ConfigParam(
            int,
            32,
            description="Number of frequency channels per chunk",
        ),
    ),
)
def load_data(upstream_output, fchunk, _cli_args_):
    """
    Load the Measurement Set data.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        fchunk: int
            Number of frequency channels per chunk
        _cli_args_: dict
            CLI Arguments.

    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    input_ms = _cli_args_["input"]
    logger.info(f"Will read from {input_ms} in {fchunk}-channel chunks")

    vis = load_ms(input_ms, fchunk)

    upstream_output["vis"] = vis
    return upstream_output
