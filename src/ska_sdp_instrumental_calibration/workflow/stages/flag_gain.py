import os

import dask
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.processing_tasks.gain_flagging import (
    flag_on_gains,
)

from ...data_managers.data_export import export_gaintable_to_h5parm


@ConfigurableStage(
    "flag_gain",
    configuration=Configuration(
        soltype=ConfigParam(
            str,
            "both",
            description="Solution type",
            allowed_values=["phase", "amplitude", "both"],
        ),
        mode=ConfigParam(
            str,
            "smooth",
            description="Detrending/fitting algorithm: smooth / poly",
            allowed_values=["smooth", "poly"],
        ),
        order=ConfigParam(
            int,
            3,
            description="Order of the function fitted during detrending.",
        ),
        apply_flag=ConfigParam(
            bool,
            True,
            description="Weights are applied to the gains",
        ),
        skip_cross_pol=ConfigParam(
            bool,
            True,
            description="Cross polarizations is skipped when flagging",
        ),
        max_rms=ConfigParam(
            float,
            5.0,
            description="Rms to clip outliers",
        ),
        fix_rms=ConfigParam(
            float,
            0.0,
            description="Instead of calculating rms use this value",
        ),
        max_ncycles=ConfigParam(
            int,
            5,
            description="Max number of independent flagging cycles",
        ),
        max_rms_noise=ConfigParam(
            float,
            0.0,
            description="""Do a running rms and then flag those regions
                    that have a rms higher than max_rms_noise*rms_of_rmses""",
        ),
        window_noise=ConfigParam(
            int,
            11,
            description="Window size for running rms",
        ),
        fix_rms_noise=ConfigParam(
            float,
            0.0,
            description="Instead of calculating rms of rmses use this value",
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
    """
    Performs flagging on gains and updates the weight.

    Parameters
    ----------
        upstream_output: dict
                Output from the upstream stage.
        soltype: str
            Solution type to flag. Can be "phase", "amplitude" or "both".
        mode: str, optional
            Detrending/fitting algorithm: "smooth", "poly".
            By default smooth.
        order : int
            Order of the function fitted during detrending.
            If mode=smooth these are the window of the running
            median (0=all axis).
        skip_cross_pol: bool
            Cross polarizations is skipped when flagging.
        export_gaintable: bool
            Export intermediate gain solution.
        max_rms: float, optional
            Rms to clip outliers, by default 5.
        fix_rms: float, optional
            Instead of calculating rms use this value, by default 0.
        max_ncycles: int, optional
            Max number of independent flagging cycles, by default 5.
        max_rms_noise: float, optional
            Do a running rms and then flag those regions that have a rms
            higher than max_rms_noise*rms_of_rmses.
        window_noise: int, optional
            Window size for the running rms, by default 11.
        fix_rms_noise: float, optional
            Instead of calculating rms of the rmses use this value
            (it will not be multiplied by the max_rms_noise), by default 0.
        apply_flag: bool
            Weights are applied to the gains.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

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
