import os

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.processing_tasks.gain_flagging import flag_on_gains
from ...data_managers.data_export import export_gaintable_to_h5parm

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
        apply_flag=ConfigParam(
            bool,
            True,
            description="cycles",
        ),
        skip_cross_pol=ConfigParam(
            bool,
            True,
            description="",
        ),
        max_rms=ConfigParam(
            float,
            5.,
            description="max rms",
        ),
        fix_rms=ConfigParam(
            float,
            0.0,
            description="fix ms",
        ),
        max_ncycles=ConfigParam(
            int,
            5,
            description="cycles",
        ),
        max_rms_noise=ConfigParam(
            float,
            0.,
            description="cycles",
        ),
        window_noise=ConfigParam(
            int,
            11,
            description="cycles",
        ),
        fix_rms_noise=ConfigParam(
            float,
            0.,
            description="cycles",
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),

    ),
)
def flag_gain_stage(
    upstream_output,
    soltype,
    mode,
    order,
    skip_cross_pol,
    export_gaintable,
    max_rms,
    fix_rms,
    max_ncycles,
    max_rms_noise,
    window_noise,
    fix_rms_noise,
    apply_flag,
    _output_dir_,
):

    initialtable = upstream_output.gaintable

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("gain_flag"):
        call_counter_suffix = f"_{call_count}"

    gaintable = flag_on_gains(
        initialtable,
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        skip_cross_pol,
        apply_flag,
    )

    if export_gaintable:
        gaintable_file_path = os.path.join(
            _output_dir_, f"gain_flag{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("gain_flag")
    return upstream_output
