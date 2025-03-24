import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "complex_gain",
    configuration=Configuration(
        config1=ConfigParam(
            int,
            2,
            description="config1",
        ),
        config2=ConfigParam(
            int,
            2,
            description="config2",
        ),
    ),
)
def complex_gain(upstream_output, config1, config2):
    """
    Perform complex gain.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    logger.info("Performing complex gain")

    upstream_output["complex_gains"] = [config1, config2]
    return upstream_output
