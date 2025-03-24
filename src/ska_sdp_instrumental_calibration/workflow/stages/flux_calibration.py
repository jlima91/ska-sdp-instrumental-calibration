import logging

from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

logger = logging.getLogger()


@ConfigurableStage(
    "flux_calibration",
    configuration=Configuration(
        config1=ConfigParam(
            int,
            2,
            description="config1",
        ),
    ),
)
def flux_calibration(upstream_output, config1):
    """
    Perform flux calibration in measurement set.

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
    Returns
    -------
        dict
            Updated upstream_output with the loaded visibility data
    """
    logger.info("Performing flux calibration")

    upstream_output["flux_calibrated_data"] = [config1]
    return upstream_output
