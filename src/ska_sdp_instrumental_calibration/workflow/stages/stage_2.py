import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "stage_2",
    configuration=Configuration(
        config1=ConfigParam(
            int,
            1,
            description="config of type int",
        ),
    ),
)
def stage_2(upstream_output, config1):
    """
    Perform stage 2

     Parameters
     ----------
         upstream_output: dict
             Output from the upstream stage
     Returns
     -------
         dict
             Updated upstream_output with data
    """
    logger.info("Performing stage 2")

    upstream_output["stage_2_data"] = [config1]
    return upstream_output
