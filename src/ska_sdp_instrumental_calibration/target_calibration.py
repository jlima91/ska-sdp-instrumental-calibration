from ska_sdp_piper.piper import CLIArgument, Pipeline

from ska_sdp_instrumental_calibration.scheduler import InstrumentalDaskRunner
from ska_sdp_instrumental_calibration.stages import (
    export_gaintable_stage,
    target_calibration,
)

from . import __version__

input_arg = CLIArgument(
    "input",
    nargs="+",
    type=str,
    help="Input visibility path",
)

ska_sdp_instrumental_target_calibration = Pipeline(
    "ska_sdp_instrumental_target_calibration",
    target_calibration.load_data_stage,
    target_calibration.predict_vis_stage,
    target_calibration.complex_gain_calibration_stage,
    export_gaintable_stage,
    version=__version__,
).overide_run(
    input_arg,
    runner=InstrumentalDaskRunner,
)

ska_sdp_instrumental_target_ionospheric_calibration = Pipeline(
    "ska_sdp_instrumental_target_ionospheric_calibration",
    target_calibration.load_data_stage,
    target_calibration.predict_vis_stage,
    target_calibration.ionospheric_delay_stage,
    export_gaintable_stage,
    version=__version__,
).overide_run(
    input_arg,
    runner=InstrumentalDaskRunner,
)
