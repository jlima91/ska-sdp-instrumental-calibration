from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..plot import plot_curve_fit, plot_flag_gain
from ..xarray_processors.gain_flagging import (
    flag_on_gains,
    log_flaging_statistics,
)
from ._utils import get_gaintables_path, get_plots_path
from .configuration_models import PlotFlagGainConfig


@ConfigurableStage(name="flag_gain", optional=True)
def flag_gain_stage(
    _upstream_output_,
    _output_dir_,
    plot_config: Annotated[
        PlotFlagGainConfig,
        Field(description="Plot options", default_factory=PlotFlagGainConfig),
    ],
    soltype: Annotated[
        Literal["phase", "amplitude", "amp-phase", "real-imag"],
        Field(
            description=(
                "Solution type. There is a potential edge case "
                "where cyclic phases my get flagged as outliers. "
                "eg -180 and 180"
            )
        ),
    ] = "amp-phase",
    order: Annotated[
        int,
        Field(description="Order of the function fitted during detrending."),
    ] = 3,
    apply_flag: Annotated[
        bool,
        Field(description="Weights are applied to the gains"),
    ] = True,
    skip_cross_pol: Annotated[
        bool,
        Field(description="Cross polarizations is skipped when flagging"),
    ] = True,
    max_ncycles: Annotated[
        int,
        Field(description="Max number of independent flagging cycles"),
    ] = 5,
    n_sigma: Annotated[
        float,
        Field(
            description="""Flag values greated than n_simga * sigma_hat.
            Where sigma_hat is 1.4826 * MeanAbsoluteDeviation."""
        ),
    ] = 3.0,
    n_sigma_rolling: Annotated[
        float,
        Field(
            description="""Do a running rms and then flag those regions
            that have a rms higher than n_sigma_rolling*MAD(rmses)."""
        ),
    ] = 10.0,
    window_size: Annotated[
        int,
        Field(description="Window size for running rms"),
    ] = 11,
    normalize_gains: Annotated[
        bool,
        Field(
            description="Normailize the amplitude and phase before flagging."
        ),
    ] = True,
    export_gaintable: Annotated[
        bool,
        Field(description="Export intermediate gain solutions."),
    ] = False,
):
    """
    Performs flagging on gains and updates the weight.

    Parameters
    ----------
         _upstream_output_: dict
            Output from the upstream stage
        _output_dir_ : str
            Directory path where the output file will be written.
        plot_config: PlotConfig
            Plotting options.
        soltype: str
            Solution type to flag.
            Can be "real-imag", "phase", "amplitude" or "amp-phase".
            There is a potential edge case
            where cyclic phases my get flagged as outliers. eg -180 and 180
        order : int
            Order of the function fitted during detrending.
            If mode=smooth these are the window of the running
            median (0=all axis).
        apply_flag: bool
            Weights are applied to the gains.
        skip_cross_pol: bool
            Cross polarizations is skipped when flagging.
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
        export_gaintable: bool
            Export intermediate gain solution.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    _upstream_output_.add_checkpoint_key("gaintable")
    initialtable = _upstream_output_.gaintable
    prefix = _upstream_output_.ms_prefix

    call_counter_suffix = ""
    if call_count := _upstream_output_.get_call_count("gain_flag"):
        call_counter_suffix = f"_{call_count}"

    gaintable, fits = flag_on_gains(
        initialtable,
        soltype,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        normalize_gains,
        skip_cross_pol,
        apply_flag,
    )

    _upstream_output_.add_compute_tasks(
        log_flaging_statistics(
            gaintable.weight,
            initialtable.weight,
        )
    )

    if plot_config.gain_flag_plot:
        path_prefix = get_plots_path(
            _output_dir_, f"{prefix}_gain_flagging{call_counter_suffix}"
        )
        _upstream_output_.add_compute_tasks(
            plot_flag_gain(
                gaintable,
                path_prefix,
                figure_title="Gain Flagging",
            )
        )

    if plot_config.curve_fit_plot:
        path_prefix = get_plots_path(
            _output_dir_, f"{prefix}_curve_fit_gain{call_counter_suffix}"
        )

        _upstream_output_.add_compute_tasks(
            plot_curve_fit(
                gaintable,
                fits,
                soltype,
                path_prefix,
                normalize_gains,
                figure_title="Curve fit of Gain Flagging",
            )
        )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_,
            f"{prefix}_gain_flag{call_counter_suffix}.gaintable.h5parm",
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    _upstream_output_["gaintable"] = gaintable
    _upstream_output_.increment_call_count("gain_flag")

    return _upstream_output_
