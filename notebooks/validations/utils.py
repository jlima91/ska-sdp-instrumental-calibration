import h5py
import matplotlib.pyplot as plt
import numpy as np

# Expected (simulated) gaintable has one extra channel at the end
# Need to remove it before comparing
REMOVE_LAST_ITEM = slice(None, -1, None)


def get_uv_wave(uvw, frequency):
    c = 3e8
    wavelength = c / frequency
    uvw_t = uvw.transpose("spatial", "time", "baselineid")
    return ((uvw_t[0] ** 2 + uvw_t[1] ** 2) ** 0.5) / wavelength


def plot_amp_uv_wave(input_vis, model_vis, prefix_path):
    fig = plt.figure(layout="constrained", figsize=(10, 5))
    fig.suptitle("Amp vs UVWave", fontsize=16)
    input_fig, model_fig = fig.subplots(1, 2)

    input_fig.set_ylim(0, 100)
    input_fig.set_title("Input visibilities")
    input_fig.set_xlabel("UVwave (λ)")
    input_fig.set_ylabel("amp")
    input_fig.scatter(
        abs(
            get_uv_wave(input_vis.uvw, input_vis.frequency).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        abs(
            input_vis.vis.isel(polarisation=0).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        s=1.0,
    )

    model_fig.set_ylim(0, 100)
    model_fig.set_title("Inst Predicted Model visibilitites")
    model_fig.set_xlabel("UVwave (λ)")
    model_fig.set_ylabel("amp")
    model_fig.scatter(
        abs(
            get_uv_wave(model_vis.uvw, model_vis.frequency).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        abs(
            model_vis.vis.isel(polarisation=0).stack(
                flatted_dim=("time", "baselineid", "frequency")
            )
        ),
        s=1.0,
    )

    fig.savefig(f"{prefix_path}/amp-uvwave.png")
    plt.close(fig)


## Amp vs Channel


def plot_amp_freq(
    input_vis, model_vis, time_step, start_baseline, end_baseline, prefix_path
):
    fig = plt.figure(layout="constrained", figsize=(10, 5))
    fig.suptitle("Amp vs Frequency", fontsize=16)
    xx_ax, yy_ax = fig.subplots(1, 2)

    xx_ax.set_title("Model XX")
    xx_ax.set_xlabel("Channel")
    xx_ax.set_ylabel("Amp")

    yy_ax.set_title("Model YY")
    yy_ax.set_xlabel("Channel")
    yy_ax.set_ylabel("Amp")
    baselines = input_vis.baselineid.values

    for i in range(start_baseline, end_baseline):
        xx_ax.plot(
            abs(model_vis.vis.isel(time=time_step, baselineid=i, polarisation=0)),
            label=baselines[i],
        )
        yy_ax.plot(
            abs(model_vis.vis.isel(time=time_step, baselineid=i, polarisation=3)),
            label=baselines[i],
        )

    handles, labels = xx_ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Baselines", loc="outside center right")
    fig.savefig(f"{prefix_path}/amp-freq.png")

    plt.close(fig)


class H5ParmIO:
    @staticmethod
    def get_frequency(h5parm_path):
        with h5py.File(h5parm_path) as act_gain_f:
            frequency = act_gain_f["sol000"]["amplitude000"]["freq"][:]
            return frequency

    @staticmethod
    def get_polarisations(h5parm_path):
        with h5py.File(h5parm_path) as act_gain_f:
            pols = act_gain_f["sol000"]["amplitude000"]["pol"][:]
            pols = [item.decode("utf-8") for item in pols]
            return pols

    @staticmethod
    def get_antennas(h5parm_path):
        with h5py.File(h5parm_path) as act_gain_f:
            stations = act_gain_f["sol000"]["amplitude000"]["ant"][:]
            stations = [item.decode("utf-8") for item in stations]
            return stations

    @staticmethod
    def get_values(
        h5parm_path,
        solset="amplitude000",
        time: slice = slice(None),
        antenna: slice = slice(None),
        frequency: slice = slice(None),
        pol: slice = slice(None),
    ):
        with h5py.File(h5parm_path) as act_gain_f:
            vals = act_gain_f["sol000"][solset]["val"][time, antenna, frequency, pol][:]
            return vals
