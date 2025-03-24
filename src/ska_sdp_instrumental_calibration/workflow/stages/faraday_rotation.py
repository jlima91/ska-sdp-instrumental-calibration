import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "faraday_rotation",
    configuration=Configuration(
        config1=ConfigParam(
            int,
            2,
            description="config1",
        ),
    ),
)
def faraday_rotation(upstream_output, config1):
    """
    Perform faraday rotation.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    logger.info("Performing faraday rotation")

    upstream_output["faraday_rotation_result"] = [config1]
    return upstream_output
