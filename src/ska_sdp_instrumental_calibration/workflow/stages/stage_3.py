import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "stage_3",
    configuration=Configuration(
        config1=ConfigParam(
            str,
            "config1",
            description="config of type string",
        ),
    ),
)
def stage_3(upstream_output, config1):
    """
    Perform stage 3

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
    Returns
    -------
        dict
            Updated upstream_output with data
    """
    logger.info("Performing stage 3")

    upstream_output["stage_3_data"] = [config1]
    return upstream_output
