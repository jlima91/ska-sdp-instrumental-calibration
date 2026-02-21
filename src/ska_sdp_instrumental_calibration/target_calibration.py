from ska_sdp_piper.piper.command import CLIArgument
from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.pipeline import Pipeline
from ska_sdp_piper.piper.stage import Stages

from ska_sdp_instrumental_calibration.scheduler import DefaultScheduler
from ska_sdp_instrumental_calibration.stages import (
    export_gaintable_stage,
    target_calibration,
)

scheduler = DefaultScheduler()


ska_sdp_instrumental_target_calibration = Pipeline(
    "ska_sdp_instrumental_target_calibration",
    stages=Stages(
        [
            target_calibration.load_data_stage,
            target_calibration.predict_vis_stage,
            target_calibration.complex_gain_calibration_stage,
            export_gaintable_stage,
        ]
    ),
    scheduler=scheduler,
    global_config=Configuration(),
    cli_args=[
        CLIArgument(
            "--input",
            dest="input",
            type=str,
            required=True,
            help="Input visibility path",
        )
    ],
    include_input_opt=False,
)

ska_sdp_instrumental_target_ionospheric_calibration = Pipeline(
    "ska_sdp_instrumental_target_ionospheric_calibration",
    stages=Stages(
        [
            target_calibration.load_data_stage,
            target_calibration.predict_vis_stage,
            target_calibration.ionospheric_delay_stage,
            export_gaintable_stage,
        ]
    ),
    scheduler=scheduler,
    global_config=Configuration(),
    cli_args=[
        CLIArgument(
            "--input",
            dest="input",
            type=str,
            required=True,
            help="Input visibility path",
        )
    ],
    include_input_opt=False,
)
