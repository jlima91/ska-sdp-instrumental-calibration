import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.processing_tasks.gain_flagging import (
    flag_on_gains,
    log_flaging_statistics,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    get_gaintables_path,
    get_plots_path,
    plot_curve_fit,
    plot_flag_gain,
)

from ...data_managers.data_export import export_gaintable_to_h5parm


@ConfigurableStage(
    "flag_gain",
    configuration=Configuration(
        soltype=ConfigParam(
            str,
            "both",
            description=(
                "Solution type. There is a potential edge case"
                " where cyclic phases my get flagged as outliers. "
                "eg -180 and 180"
            ),
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
        max_ncycles=ConfigParam(
            int,
            5,
            description="Max number of independent flagging cycles",
        ),
        n_sigma=ConfigParam(
            float,
            10.0,
            description="""Flag values greated than n_simga * sigma_hat.
            Where sigma_hat is 1.4826 * MeanAbsoluteDeviation.""",
        ),
        n_sigma_rolling=ConfigParam(
            float,
            10.0,
            description="""Do a running rms and then flag those regions
            that have a rms higher than n_sigma_rolling*MAD(rmses).""",
        ),
        window_size=ConfigParam(
            int,
            11,
            description="Window size for running rms",
        ),
        normalize_gains=ConfigParam(
            bool,
            True,
            description="Normailize the amplitude and phase before flagging.",
        ),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
        plot_config=NestedConfigParam(
            "Plot options",
            curve_fit_plot=ConfigParam(
                bool,
                True,
                description="Plot the fitted curve of gain flagging",
                nullable=False,
            ),
            gain_flag_plot=ConfigParam(
                bool,
                True,
                description="Plot the flagged weights",
                nullable=False,
            ),
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
    max_ncycles,
    n_sigma,
    n_sigma_rolling,
    window_size,
    normalize_gains,
    apply_flag,
    plot_config,
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
            There is a potential edge case
            where cyclic phases my get flagged as outliers. eg -180 and 180
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
        max_ncycles: int, optional
            Max number of independent flagging cycles, by default 5.
        n_sigma: float, optional
            Flag values greated than n_simga * sigma_hat.
            Where sigma_hat is 1.4826 * MeanAbsoluteDeviation
            Defaulted to 10
        n_sigma_rolling: float, optional
            Do a running rms and then flag those regions that have a rms
            higher than n_sigma_rolling*MAD(rmses).
            Defaulted to 5
        window_size: int, optional
            Window size for the running rms, by default 11.
        normalize_gains: bool
            Normailize the amplitude and phase before flagging.
        apply_flag: bool
            Weights are applied to the gains.
        plot_config: dict
            Plotting options.
            eg: {curve_fit_plot: True, gain_flag_plot: True}

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    upstream_output.add_checkpoint_key("gaintable")
    initialtable = upstream_output.gaintable

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("gain_flag"):
        call_counter_suffix = f"_{call_count}"

    gaintable, amp_fit, phase_fits = flag_on_gains(
        initialtable,
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        skip_cross_pol,
        apply_flag,
    )

    upstream_output.add_compute_tasks(
        log_flaging_statistics(
            gaintable.weight,
            initialtable.weight,
            gaintable.configuration.names.data,
        )
    )

    if plot_config["gain_flag_plot"]:
        path_prefix = get_plots_path(
            _output_dir_, f"gain_flagging{call_counter_suffix}"
        )
        upstream_output.add_compute_tasks(
            plot_flag_gain(
                gaintable,
                path_prefix,
                figure_title="Gain Flagging",
            )
        )

    if plot_config["curve_fit_plot"]:
        path_prefix = get_plots_path(
            _output_dir_, f"curve_fit_gain{call_counter_suffix}"
        )

        upstream_output.add_compute_tasks(
            plot_curve_fit(
                gaintable,
                amp_fit,
                phase_fits,
                path_prefix,
                figure_title="Curve fit of Gain Flagging",
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
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
