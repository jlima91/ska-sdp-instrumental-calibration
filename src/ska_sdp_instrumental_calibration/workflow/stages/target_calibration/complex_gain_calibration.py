import logging
import os
from copy import deepcopy

import dask
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm as h5exp,
)
from ska_sdp_instrumental_calibration.processing_tasks.calibrate import (
    target_solver,
)
from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages._common import (
    RUN_SOLVER_DOCSTRING,
    RUN_SOLVER_NESTED_CONFIG,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    parse_reference_antenna,
)

logger = logging.getLogger()


@ConfigurableStage(
    "complex_gain_calibration",
    configuration=Configuration(
        run_solver_config=deepcopy(RUN_SOLVER_NESTED_CONFIG),
        visibility_key=ConfigParam(
            str,
            "vis",
            description="Visibility data to be used for calibration.",
            allowed_values=["vis", "corrected_vis"],
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def complex_gain_calibration_stage(
    upstream_output: UpstreamOutput,
    run_solver_config,
    visibility_key,
    export_gaintable,
    _output_dir_,
):
    """
    Performs Complex Gain Calibration

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage. It should contain:
              gaintable, modelvis and visibility data with key
              same as visibility_key
        run_solver_config: dict
            {run_solver_docstring}
        visibility_key: str
            Visibility data to be used for calibration.
        export_gaintable: bool
            Export intermediate gain solutions
        _output_dir_ : str
            Directory path where the output file will be written.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    upstream_output.add_checkpoint_key("gaintable")
    modelvis = upstream_output.modelvis
    initial_gaintable = upstream_output.gaintable

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for complex gain calibration.")

    refant = run_solver_config["refant"]
    run_solver_config["refant"] = parse_reference_antenna(
        refant, initial_gaintable
    )

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("complex_gain"):
        call_counter_suffix = f"_{call_count}"

    gaintable = target_solver.run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initial_gaintable,
        **run_solver_config,
    )

    if export_gaintable:
        gaintable_file_path = os.path.join(
            _output_dir_, f"complex_gain{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(h5exp.export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("complex_gain")
    return upstream_output


complex_gain_calibration_stage.__doc__ = (
    complex_gain_calibration_stage.__doc__.format(
        run_solver_docstring=RUN_SOLVER_DOCSTRING
    )
)
