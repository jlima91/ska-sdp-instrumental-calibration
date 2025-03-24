import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "delay_calibration",
    configuration=Configuration(
        config1=ConfigParam(
            int,
            1,
            description="Config1",
        ),
    ),
)
def delay_calibration(upstream_output, config1):
    """
    Perform delay calibration in measurement set.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    logger.info("Performing delay calibration")

    upstream_output["delay_calibrated_data"] = [config1]
    return upstream_output
