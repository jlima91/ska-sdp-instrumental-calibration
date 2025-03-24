import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "flag_ms",
    configuration=Configuration(
        config1=ConfigParam(
            int,
            1,
            description="some config of type int",
        ),
    ),
)
def flag_ms(upstream_output, config1):
    """
    Flag the bad data in measurement set.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    logger.info("Flagging data")

    upstream_output["flagged_data"] = [config1]
    return upstream_output
