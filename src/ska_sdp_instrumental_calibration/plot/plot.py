import dask.delayed
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    generate_rotation_matrices,
)

from ..workflow.utils import normalize_data
from ._util import _ecef_to_lla, safe
from .plot_gaintable import PlotGaintableFrequency

matplotlib.use("Agg")

logger = setup_logger(__name__)

__all__ = [
    "plot_station_delays",
    "plot_rm_station",
    "plot_bandpass_stages",
    "plot_flag_gain",
    "plot_curve_fit",
]


@dask.delayed
@safe
def plot_flag_gain(
    gaintable,
    path_prefix,
    figure_title="",
):
    gaintable = gaintable.stack(pol=("receptor1", "receptor2"))

    polstrs = [f"{p1}{p2}".upper() for p1, p2 in gaintable.pol.data]
    gaintable = gaintable.assign_coords({"pol": polstrs})
    stations = gaintable.configuration.names

    n_rows = 4
    n_cols = 4
    plots_per_group = n_rows * n_cols
    plot_groups = np.split(
        stations,
        range(plots_per_group, stations.size, plots_per_group),
    )

    for stations in plot_groups:
        station_names = stations.values
        fig = plt.figure(layout="constrained", figsize=(18, 18))
        subfigs = fig.subfigures(n_rows, n_cols).reshape(-1)
        primary_axes = None

        for idx, subfig in enumerate(subfigs):
            if idx >= stations.size:
                break
            weight = gaintable.weight.isel(
                time=0, antenna=stations.id[idx], pol=0
            )
            weight_ax = subfig.subplots(1, 1, sharex=True)
            primary_axes = weight_ax
            weight_ax.set_ylabel("Weights")
            weight_ax.set_xlabel("Channel")
            weight_ax.plot(weight)
            subfig.suptitle(
                f"Station - {station_names[idx]}", fontsize="large"
            )

        handles, labels = primary_axes.get_legend_handles_labels()
        path = (
            f"{path_prefix}-weights_freq-"
            f"{station_names[0]}-{station_names[-1]}.png"
        )
        fig.suptitle(f"{figure_title}", fontsize="x-large")
        fig.legend(handles, labels, loc="outside upper right")
        fig.savefig(path)
        plt.close()


@dask.delayed
@safe
def plot_curve_fit(
    gaintable,
    amp_fits,
    phase_fits,
    path_prefix,
    normalize_gains=False,
    figure_title="",
):
    """
    Plot the fitted curve on gains

    Parameters
    ----------
        gaintable: xr.Dataset
            Gaintable to plot.
        amp_fits: xr.DataArray.
            Amplitude fits.
        phase_fits: xr.DataArray.
            Phase fits.
        path_prefix: str
            Path prefix to save the plots.
        figure_title: str
            Title of the figure.
        normalize_gains: bool
            Plot for normalized gains.
        fixed_axis: bool
            Limit amplitude axis values to [0,1]
    """

    normalize_label = "(normalized)" if normalize_gains else ""
    gaintable = gaintable.stack(pol=("receptor1", "receptor2"))
    amp_fits = amp_fits.stack(pol=("receptor1", "receptor2"))
    phase_fits = phase_fits.stack(pol=("receptor1", "receptor2"))

    # from SKB-1027. J_XX, J_YY, j_xy and j_yx
    polstrs = [f"J_{p1}{p2}".upper() for p1, p2 in gaintable.pol.data]

    gaintable = gaintable.assign_coords({"pol": polstrs})
    amp_fits = amp_fits.assign_coords({"pol": polstrs})
    phase_fits = phase_fits.assign_coords({"pol": polstrs})
    stations = gaintable.configuration.names
    n_rows = 2
    n_cols = 2
    plots_per_group = n_rows * n_cols
    plot_groups = np.split(
        stations,
        range(plots_per_group, stations.size, plots_per_group),
    )

    gain = gaintable.gain.isel(time=0)

    pol_labels = gaintable.pol.values
    frequency = gaintable.frequency / 1e6
    channel = np.arange(len(frequency))

    pol_groups = np.array(polstrs)[[0, 3, 1, 2]].reshape(
        -1, 2
    )  # [['J_XX', 'J_YY'],['J_XY', 'J_YX']]
    normalize_func = normalize_data if normalize_gains else lambda x: x
    plot_freq = PlotGaintableFrequency()

    for stations in plot_groups:
        station_names = stations.values

        cmap = plt.get_cmap("tab10")
        pol_colors = {pol: cmap(i) for i, pol in enumerate(pol_labels)}

        fig = plt.figure(layout="constrained", figsize=(24, 18))
        subfigs = fig.subfigures(n_rows, n_cols).reshape(-1)

        scatter_kwargs = dict(alpha=0.4, s=15)
        plot_kwargs = dict(lw=2)

        path = (
            f"{path_prefix}-curve-amp-phase_freq-"
            f"{station_names[0]}-{station_names[-1]}.png"
        )

        fig.suptitle(
            f"{figure_title} Solutions {normalize_label}", fontsize="x-large"
        )

        all_handles = []
        all_labels = []

        for idx, subfig in enumerate(subfigs):
            if idx >= stations.size:
                break
            # gain = gaintable.gain.isel(time=0, antenna=stations.id[idx])
            a_fit = amp_fits.isel(time=0, antenna=stations.id[idx])
            p_fit = np.rad2deg(
                phase_fits.isel(time=0, antenna=stations.id[idx])
            )

            amplitude = np.abs(gain.isel(antenna=stations.id[idx]))
            phase = np.angle(gain.isel(antenna=stations.id[idx]), deg=True)

            axes = subfig.subplots(2, 2, sharex=True)

            for grp_idx, lbl_idx in np.ndindex(pol_groups.shape):
                phase_ax = axes[0, grp_idx]
                amp_ax = axes[1, grp_idx]
                amp_ax.set_ylabel("Amplitude")
                phase_ax.set_ylabel("Phase (degree)")

                amp_ax.set_xlabel("Channel")

                phase_ax.secondary_xaxis(
                    "top",
                    functions=(
                        plot_freq._primary_sec_ax_mapper(frequency, channel),
                        plot_freq._primary_sec_ax_mapper(
                            frequency, channel, reverse=True
                        ),
                    ),
                ).set_xlabel("Frequency [MHz]")

                phase_ax.set_ylim([-180, 180])
                pol = pol_groups[grp_idx, lbl_idx]
                if pol in pol_labels:
                    pol_idx = list(pol_labels).index(pol)
                    h1 = phase_ax.scatter(
                        channel,
                        phase[:, pol_idx],
                        color=pol_colors[pol],
                        label=pol,
                        **scatter_kwargs,
                    )
                    amp_ax.scatter(
                        channel,
                        normalize_func(amplitude[:, pol_idx].values),
                        color=pol_colors[pol],
                        label=pol,
                        **scatter_kwargs,
                    )
                    phase_ax.plot(
                        channel,
                        p_fit[:, pol_idx],
                        color=pol_colors[pol],
                        label=pol,
                        **plot_kwargs,
                    )
                    amp_ax.plot(
                        channel,
                        a_fit[:, pol_idx],
                        color=pol_colors[pol],
                        label=pol,
                        **plot_kwargs,
                    )
                    if pol not in all_labels:
                        all_handles.append(h1)
                        all_labels.append(pol)

            subfig.suptitle(
                f"Station - {station_names[idx]}", fontsize="large"
            )

        fig.legend(all_handles, all_labels, loc="outside upper right")
        fig.savefig(path)
        plt.close()


@dask.delayed
@safe
def plot_station_delays(delaytable, path_prefix):
    """
    Plot the station delays against the station configuration

    Parameters
    ----------
        delaytable: xr.Dataset
            Delay dataset
        path_prefix: str
            Path prefix to save the plots.
    """

    latitude, longitude, _ = _ecef_to_lla(*delaytable.configuration.xyz.data.T)
    calibration_delay = delaytable.delay.data / 1e-9
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    fig, subfigs = plt.subplots(figsize=(20, 10), ncols=2, nrows=2)
    station_name = delaytable.configuration.names.data
    fig.suptitle("Station Delays")
    for idx, ax in enumerate(subfigs[0]):
        sc = ax.scatter(
            longitude,
            latitude,
            c=calibration_delay[..., idx],
            cmap="plasma",
            s=10,
        )
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Delay (ns)", rotation=270, labelpad=15)
        ax.grid()
        ax.set_title(delaytable.pol.data[idx])

    for idx, ax in enumerate(subfigs[1]):
        sc = ax.plot(
            station_name,
            calibration_delay[..., idx].reshape(len(station_name)),
        )
        ax.set_xlabel("Stations")
        ax.set_ylabel("Delay (ns)")
        ax.set_title(delaytable.pol.data[idx])
        ax.tick_params(axis="x", rotation=90)

    plt.savefig(f"{path_prefix}_station_delay.png")
    plt.close()


@dask.delayed
@safe
def plot_bandpass_stages(
    gaintable, initialtable, rm_est, refant, plot_path_prefix
):
    """
    Plot RM estimates of stations

    Parameters
    ----------
        gaintable: Gaintable Dataset
            Gaintable
        initialtable: Gaintable Dataset
            Initial gaintable
        rm_est: xr.DataArray
            rm estimate array.
        refant: int
            Reference antenna
        plot_path_prefix: str
            plot prefix
    """
    x = gaintable.frequency.data / 1e6
    stns = np.abs(rm_est).argsort()[[len(rm_est) // 4, len(rm_est) // 2, -1]]
    fig, axs = plt.subplots(3, 4, figsize=(16, 16), sharey=True)

    station_names = gaintable.configuration.names.data
    ref_stn_name = station_names[refant]

    for k, stn in enumerate(stns):
        stn_name = station_names[stn]
        J = initialtable.gain.data[0, stn] @ np.linalg.inv(
            initialtable.gain.data[0, refant, ..., :, :]
        )
        ax = axs[k, 0]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            ax.plot(x, np.imag(J[:, p, p]), f"C{pol}--")
        ax.set_title(
            f"Bandpass for station {stn_name} \n(in rel to {ref_stn_name})",
            fontsize=10,
        )
        ax.grid()
        ax.legend()

        J = generate_rotation_matrices(rm_est, gaintable.frequency.data)[stn]
        ax = axs[k, 1]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            ax.plot(x, np.imag(J[:, p, p]), f"C{pol}--")
        ax.set_title(f"RM model, RM = {rm_est[stn]:.3f}")
        ax.grid()
        ax.legend()

        J = gaintable.gain.data[0, stn] @ np.linalg.inv(
            gaintable.gain.data[0, refant, ..., :, :]
        )
        ax = axs[k, 2]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            ax.plot(x, np.imag(J[:, p, p]), f"C{pol}--")
        ax.set_title("De-rotated (re: -, im: --)")
        ax.grid()
        ax.legend()

        ax = axs[k, 3]
        for pol in range(4):
            p = pol // 2
            q = pol % 2
            ax.plot(x, np.abs(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
            if p == q:
                ax.plot(x, np.angle(J[:, p, p]), f"C{pol}--")
        ax.set_title("De-rotated (abs: -, angle: --)")
        ax.grid()
        ax.legend()

    fig.savefig(f"{plot_path_prefix}-bandpass_stages.png")


@dask.delayed
@safe
def plot_rm_station(
    gaintable,
    rm_vals,
    rm_spec,
    rm_peak,
    rm_est,
    rm_est_refant,
    J,
    lambda_sq,
    xlim,
    stn,
    plot_path_prefix,
):
    """
    Plot RM estimates of stations

    Parameters
    ----------
        gaintable: Gaintable Dataset
            Gaintable
        rm_vals: xr.DataArray
            rm value array.
        rm_spec: xr.DataArray
            rm spec array.
        rm_peak: xr.DataArray
            rm peak array.
        rm_est: xr.DataArray
            rm estimate array.
        rm_est_refant: xr.DataArray
            rm estimate of refant.
        J: xr.DataArray
            Jones array.
        lambda_sq: xr.DataArray
            lambda square array.
        xlim: xr.DataArray
            x-limit array.
        stn: int
            station number.
        plot_path_prefix: str
            plot prefix
    """
    fig = plt.figure(figsize=(14, 12))

    x = gaintable.frequency.data / 1e6
    station_names = gaintable.configuration.names.data
    stn_name = station_names[stn]

    ax = fig.add_subplot(311)
    ax.plot(rm_vals, np.abs(rm_spec), "b", label="abs")
    ax.plot(rm_vals, np.real(rm_spec), "c", label="re")
    ax.plot(rm_vals, np.imag(rm_spec), "m", label="im")
    ax.plot(rm_peak * np.ones(2), ax.get_ylim(), "c-")
    ax.plot(rm_est * np.ones(2), ax.get_ylim(), "b--")
    ax.set_xlim((-xlim, xlim))
    ax.set_title(f"RM spectrum. Peak = {rm_est:.3f} (rad / m^2)")
    ax.set_xlabel("RM (rad / m^2)")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(323)
    for pol in range(4):
        p = pol // 2
        q = pol % 2
        ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
        ax.plot(x, np.imag(J[:, p // 2, p % 2]), f"C{pol}--")
    ax.set_title(f"Bandpass Jones for station {stn_name} (re: -, im: --)")
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(324)
    d_pa = (rm_est - rm_est_refant) * lambda_sq
    R = np.stack(
        (np.cos(d_pa), np.sin(d_pa), -np.sin(d_pa), np.cos(d_pa)),
        axis=1,
    ).reshape(-1, 2, 2)
    for p in range(4):
        ax.plot(x, np.real(R[:, p // 2, p % 2]), f"C{p}")
        ax.plot(x, np.imag(R[:, p // 2, p % 2]), f"C{p}--")
    ax.set_title("RM rotation matrices")
    ax.grid()

    ax = fig.add_subplot(325)
    B = J @ np.linalg.inv(R[..., :, :])
    for p in range(4):
        ax.plot(x, np.real(B[:, p // 2, p % 2]), f"C{p}")
        ax.plot(x, np.imag(B[:, p // 2, p % 2]), f"C{p}--")
    ax.set_title("De-rotated bandpass Jones (re: -, im: --)")
    ax.set_xlabel("Frequency (MHz)")
    ax.grid()

    ax = fig.add_subplot(326)
    B = J @ np.linalg.inv(R[..., :, :])
    for p in [0, 3]:
        ax.plot(x, np.abs(B[:, p // 2, p % 2]), f"C{p}")
        ax.plot(x, np.angle(B[:, p // 2, p % 2]), f"C{p}--")
    ax.set_title("De-rotated bandpass Jones (abs: -, angle: --)")
    ax.set_xlabel("Frequency (MHz)")
    ax.grid()

    fig.savefig(f"{plot_path_prefix}-rm-station-{stn_name}.png")
