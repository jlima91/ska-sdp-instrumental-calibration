from ska_sdp_piper.piper import CLIArgument, Pipeline

from ska_sdp_instrumental_calibration.scheduler import InstrumentalDaskRunner
from ska_sdp_instrumental_calibration.stages import (
    export_gaintable_stage,
    target_calibration,
)

from . import __version__
from .sdm import prepare_qa_path

input_arg = CLIArgument(
    "input_ms",
    nargs="+",
    type=str,
    help="Input visibility path(s)",
)

sdm_cli_arg = CLIArgument(
    "--sdm-path",
    dest="sdm_path",
    type=str,
    default=None,
    help="""Directory path to store the Science Data Models""",
)

ska_sdp_instrumental_target_calibration = (
    Pipeline(
        "ska_sdp_instrumental_target_calibration",
        target_calibration.load_data_stage,
        target_calibration.predict_vis_stage,
        target_calibration.complex_gain_calibration_stage,
        export_gaintable_stage,
        version=__version__,
    )
    .with_qa_path_resolver(prepare_qa_path)
    .overide_run(
        input_arg,
        sdm_cli_arg,
        runner=InstrumentalDaskRunner,
    )
)

ska_sdp_instrumental_target_ionospheric_calibration = (
    Pipeline(
        "ska_sdp_instrumental_target_ionospheric_calibration",
        target_calibration.load_data_stage,
        target_calibration.predict_vis_stage,
        target_calibration.ionospheric_delay_stage,
        export_gaintable_stage,
        version=__version__,
    )
    .with_qa_path_resolver(prepare_qa_path)
    .overide_run(
        input_arg,
        sdm_cli_arg,
        runner=InstrumentalDaskRunner,
    )
)
