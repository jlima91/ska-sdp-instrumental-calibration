import os

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

# from ska_sdp_instrumental_calibration.processing_tasks.gain_flagging import run_flagger


@ConfigurableStage(
    "flag_gain",
    configuration=Configuration(
        soltype=ConfigParam(
            str,
            "both",
            description="solution type",
            allowed_values=["phase", "amplitude", "both"],
        ),
        mode=ConfigParam(
            str,
            "smooth",
            description="Mode",
            allowed_values=["smooth", "poly"],
        ),
        order=ConfigParam(
            int,
            3,
            description="order",
        ),
        max_rms=ConfigParam(
            float,
            0.0,
            description="max rms",
        ),
        max_ncycles=ConfigParam(
            int,
            3,
            description="cycles",
        ),
    ),
)
def flag_gain_stage(
    upstream_output, soltype, mode, order, max_rms, max_ncycles
):

    initialtable = upstream_output.gaintable
    # upstream_output.gaintable = run_flagger(
    #     initialtable,
    #     soltype,
    #     mode,
    #     order,
    #     max_rms,
    #     max_ncycles
    # )

    return upstream_output
