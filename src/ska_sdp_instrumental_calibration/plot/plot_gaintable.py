import dask
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from ska_sdp_instrumental_calibration.logger import setup_logger

from ._util import safe

logger = setup_logger(__name__)


class PlotGaintable:
    """
    Base class for plotting gaintable solutions (amplitude and phase).

    This class provides the core logic for preparing gaintable data
    (an xarray.Dataset) and generating faceted scatter plots. It is
    intended to be subclassed (e.g., by `PlotGaintableFrequency` or
    `PlotGaintableTime`) to specify the x-axis dimension and its
    corresponding secondary axis.

    Parameters
    ----------
    path_prefix : str, optional
        A prefix used to construct the output filenames for the plots.
        Defaults to None.

    Attributes
    ----------
    _plot_args : dict
        Default arguments passed to the `xarray.plot.scatter` method.
    _x_label : str
        Label for the primary x-axis.
    _x_sec_label : str
        Label for the secondary x-axis (top).
    _x_data : array-like
        Data used to map between the primary and secondary x-axes.
    _x_sec_data : array-like
        Data used to map between the primary and secondary x-axes.
    _path_prefix : str
        Storage for the provided `path_prefix`.
    """

    def __init__(self, path_prefix=None):
        """
        Initialize the base gaintable plotter.

        Parameters
        ----------
        path_prefix : str, optional
            A prefix used to construct the output filenames for the plots.
            Defaults to None.
        """
        self.__plot_path = f"{path_prefix}-{{plot_type}}-{self.xdim}.png"
        self._post_title = ""
        self._plot_args = dict(
            x=None,
            hue="Jones_Solutions",
            col="Station",
            col_wrap=5,
            add_legend=True,
            add_colorbar=False,
            sharex=False,
            edgecolors="none",
            aspect=1.5,
            s=8,
        )

        self._x_label = None
        self._x_sec_label = None
        self._x_data = None
        self._x_sec_data = None
        self._path_prefix = path_prefix

    @property
    def xdim(self):
        """
        str: Abstract property for the x-dimension name.

        This property defines a short string used in the output plot filename
        to identify the x-axis dimension (e.g., 'time' or 'freq').

        Raises
        ------
        NotImplementedError
            This property must be overridden by a subclass.
        """
        raise NotImplementedError("xdim property not implemented")

    def observation_start_time(self, gaintable):
        """
        Extract the observation start time from the gaintable.

        Converts the first time timestamp in the gaintable (assumed to be
        seconds) to a UTC datetime object.

        Parameters
        ----------
        gaintable : xarray.Dataset
            The input gaintable containing a 'time' coordinate.

        Returns
        -------
        numpy.datetime64
            The start time of the observation in UTC.
        """
        return Time(
            gaintable.time[0] / 86400.0, format="mjd", scale="utc"
        ).datetime64

    @dask.delayed
    @safe
    def plot(
        self,
        gaintable,
        figure_title="",
        drop_cross_pols=False,
        fixed_axis=False,
        phase_only=False,
        plot_all_stations=False,
    ):
        """
        Generate and save facet plots for gaintable phase and amplitude.

        This is a Dask delayed method that creates two plots (unless
        `phase_only` is True): one for the phase and one for the amplitude
        of the gains, faceted by station and colored by Jones solution.
        The plots are saved to disk using the `path_prefix`.

        Parameters
        ----------
        gaintable : xarray.Dataset
            The input gaintable dataset to plot.
        figure_title : str, optional
            A prefix for the main figure title. Defaults to "".
        drop_cross_pols : bool, optional
            If True, cross-polarization solutions (e.g., J_XY, J_YX) are
            dropped before plotting. Defaults to False.
        fixed_axis : bool, optional
            If True, the y-axis for the amplitude plot is fixed
            between 0 and 1. Defaults to False.
        phase_only : bool, optional
            If True, only the phase plot is generated and saved.
            Defaults to False.
        plot_all_stations : bool, optional
            If True, calls the `_plot_all_stations` method to generate
            an additional overview plot. Defaults to False.

        Returns
        -------
        None
        """
        gaintable = self._prepare_gaintable(gaintable, drop_cross_pols)
        gain_phase = gaintable.gain.copy()
        gain_phase.data = np.angle(gaintable.gain, deg=True)
        ylim = (-180, 180) if fixed_axis else None
        gain_phase_fig = self._get_gain_facet(
            gain_phase, ylim, "Phase (degree)"
        ).fig

        gain_phase_fig.suptitle(
            f"{figure_title} Solutions (Phase){self._post_title}",
            fontsize="x-large",
            y=1.08,
        )
        gain_phase_fig.tight_layout()
        gain_phase_fig.savefig(
            self.__plot_path.format(plot_type="phase"), bbox_inches="tight"
        )

        if not phase_only:
            gain_amplitude = np.abs(gaintable.gain)
            ylim = (0, 1) if fixed_axis else None
            gain_amp_fig = self._get_gain_facet(
                gain_amplitude,
                ylim,
                "Amplitude",
            ).fig

            gain_amp_fig.suptitle(
                f"{figure_title} Solutions (Amplitude){self._post_title}",
                fontsize="x-large",
                y=1.08,
            )
            gain_amp_fig.tight_layout()
            gain_amp_fig.savefig(
                self.__plot_path.format(plot_type="amp"), bbox_inches="tight"
            )

        plt.close()

        if plot_all_stations:
            self._plot_all_stations(gaintable)

    def _primary_sec_ax_mapper(self, map_from, map_to, reverse=False):
        """
        Create a mapping function between primary and secondary axes.

        Abstract function to be implemented by subclasses to handle the
        interpolation or calculation required to map x-axis values (e.g.,
        Frequency to Channel ID).

        Parameters
        ----------
        map_from : array-like
            The array of data to map from.
        map_to : array-like
            The array of data to map to.
        reverse : bool, optional
            If False returns direct mapper. If True returns the inverse map.
            Defaults to False.

        Returns
        -------
        callable
            The interpolation function.

        Raises
        ------
        NotImplementedError
            If not overridden by subclass.
        """
        raise NotImplementedError("_primary_sec_ax_mapper not defined")

    def _update_facet(self, facet_plot, y_label):
        """
        Update facet plot labels and add a secondary x-axis.

        This method formats the facet plot, setting labels and font sizes,
        and adds a secondary x-axis to the top of each subplot using the
        mapping functions defined in the subclass.

        Parameters
        ----------
        facet_plot : xarray.plot.FacetGrid
            The FacetGrid object to modify.
        y_label : str
            The label to apply to the y-axis of the first column.
        """
        for ax in facet_plot.axs[:, 0]:
            ax.set_ylabel(y_label, fontsize="small")

        for ax in facet_plot.axs.flat:
            ax.set_xlabel(self._x_label, fontsize="small")
            sec_ax = ax.secondary_xaxis(
                "top",
                functions=(
                    self._primary_sec_ax_mapper(
                        self._x_data, self._x_sec_data
                    ),
                    self._primary_sec_ax_mapper(
                        self._x_data, self._x_sec_data, reverse=True
                    ),
                ),
            )
            sec_ax.set_xlabel(self._x_sec_label, fontsize="small")
            sec_ax.tick_params(axis="x", labelsize=8)

    def _plot_all_stations(self, gaintable):
        """
        Generate a summary plot of all stations on a single figure.

        Abstract method.

        Parameters
        ----------
        gaintable : xarray.Dataset
            The processed gaintable dataset.

        Raises
        ------
        NotImplementedError
            This method must be overridden by a subclass.
        """
        raise NotImplementedError("plot_all_stations not implemented")

    def _get_gain_facet(self, gain_component, y_lim, y_label):
        """
        Create a facet grid scatter plot for a specific gain component.

        Parameters
        ----------
        gain_component : xarray.DataArray
            The gain data (e.g., amplitude or phase) to plot.
        y_lim : tuple or None
            Desired y-axis limits (e.g., `(-180, 180)`).
        y_label : str
            The label for the y-axis.

        Returns
        -------
        xarray.plot.FacetGrid
            The generated FacetGrid object.
        """
        facet_plot = gain_component.plot.scatter(**self._plot_args, ylim=y_lim)

        facet_plot.set_axis_labels(self._x_label, y_label)
        for ax in facet_plot.axs.reshape(-1):
            ax.tick_params(labelbottom=True)

        self._update_facet(facet_plot, y_label)
        facet_plot.set_ticks(fontsize=8)

        return facet_plot

    def _prepare_gaintable(self, gain_table, drop_cross_pols=False):
        """
        Prepare the gaintable xarray.Dataset for plotting.

        This method stacks the receptor dimensions into a single
        'Jones_Solutions' dimension, renames the coordinates to
        strings (e.g., 'J_XX'), adds a 'Station' coordinate,
        and optionally filters out cross-polarizations.

        Parameters
        ----------
        gain_table : xarray.Dataset
            The raw gaintable dataset to process.
        drop_cross_pols : bool, optional
            If True, cross-polarization solutions (e.g., J_XX, J_YY) are
            dropped. Defaults to False.

        Returns
        -------
        xarray.Dataset
            The processed gaintable ready for plotting.
        """
        gaintable = gain_table.stack(
            Jones_Solutions=("receptor1", "receptor2")
        )
        polstrs = [
            f"J_{p1}{p2}".upper()
            for p1, p2 in gaintable["Jones_Solutions"].data
        ]
        gaintable = gaintable.assign_coords({"Jones_Solutions": polstrs})
        gaintable.coords["Station"] = (
            "antenna",
            gaintable.configuration.names.data,
        )

        if drop_cross_pols:
            gaintable = gaintable.sel(Jones_Solutions=["J_XX", "J_YY"])

        return gaintable.swap_dims({"antenna": "Station"})


class PlotGaintableFrequency(PlotGaintable):
    """
    Plots gaintable solutions against frequency/channel.

    This class extends `PlotGaintable` to handle plotting where the primary
    x-axis represents 'Channel' (index) and the secondary x-axis represents
    'Frequency [MHz]'. It interpolates between these two domains to provide
    dual-axis visualizations.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the `PlotGaintable` parent class, such
        as `path_prefix`.
    """

    def __init__(self, **kwargs):
        """
        Initialize the frequency-based gaintable plotter.

        Sets the primary x-axis label to 'Channel' and the secondary x-axis
        label to 'Frequency [MHz]'.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the `PlotGaintable` parent class.
        """
        super(PlotGaintableFrequency, self).__init__(**kwargs)
        self._plot_args["x"] = "Channel"
        self._x_label = "Channel"
        self._x_sec_label = "Frequency [MHz]"

    def _prepare_gaintable(self, gain_table, drop_cross_pols=False):
        """
        Prepare the gaintable for frequency-axis plotting.

        Calls the parent preparation method and then adds a 'Channel'
        coordinate dimension by mapping the 'frequency' dimension to integer
        indices. It stores the frequency values (converted to MHz) in
        `self._x_data` to facilitate axis mapping.

        Parameters
        ----------
        gain_table : xarray.Dataset
            The raw gaintable dataset to process.
        drop_cross_pols : bool, optional
            If True, cross-polarization solutions are dropped.
            Defaults to False.

        Returns
        -------
        xarray.Dataset
            The processed gaintable with a 'Channel' coordinate, with
            dimensions swapped such that 'Channel' is the active dimension.
        """
        gaintable = super(PlotGaintableFrequency, self)._prepare_gaintable(
            gain_table, drop_cross_pols
        )
        self._x_data = gaintable.frequency / 1e6
        self._x_sec_data = np.arange(len(self._x_data))
        gaintable.coords["Channel"] = (
            "frequency",
            np.arange(len(gaintable.frequency)),
        )

        return gaintable.swap_dims({"frequency": "Channel"})

    @property
    def xdim(self):
        """
        str: Specifies the x-dimension name for frequency plots.

        Returns
        -------
        str
            Returns "freq".
        """
        return "freq"

    def _primary_sec_ax_mapper(self, frequency, channel, reverse=False):
        """
        Create an interpolation function for axis mapping.

        Generates a function to map between channel indices and frequency
        values (in MHz) using linear interpolation. This is used by the
        plotting engine to render the secondary top axis.

        Parameters
        ----------
        frequency : array-like
            The array of frequency values (in MHz).
        channel: array-like
            The array of channel indices.
        reverse : bool, optional
            If False (default), returns a function that maps
            Channel index -> Frequency.
            If True, returns a function that maps
            Frequency -> Channel index.

        Returns
        -------
        callable
            The interpolation function.
        """
        if reverse:
            return lambda freq: np.interp(freq, frequency, channel)

        return lambda ch: np.interp(ch, channel, frequency)

    def _plot_all_stations(self, gaintable):
        """
        Plot amplitude vs. frequency for all stations on one figure.

        Generates a summary line plot showing the amplitude of the J_XX and
        J_YY solutions (at the first time-slice) against frequency. Each
        station is represented by a different colored line. The resulting
        figure is saved to disk using the configured path prefix.

        Parameters
        ----------
        gaintable : xarray.Dataset
            The processed gaintable dataset (must have 'Channel' coord).
        """
        amplitude = np.abs(
            gaintable.gain.sel(Jones_Solutions=["J_XX", "J_YY"]).isel(time=0)
        ).swap_dims({"Channel": "frequency"})
        facet_plot = amplitude.plot.line(
            x="frequency",
            col="Jones_Solutions",
            hue="Station",
            figsize=(18, 10),
        )
        facet_plot.set_axis_labels("Freq [HZ]", "Amplitude")

        facet_plot.fig.savefig(
            f"{self._path_prefix}-all_station_amp_vs_freq.png",
            bbox_inches="tight",
        )
        plt.close(facet_plot.fig)


class PlotGaintableTime(PlotGaintable):
    """
    Plots gaintable solutions against time.

    This class extends `PlotGaintable` to handle plotting where the primary
    x-axis represents 'Observation Time (S)' (relative to the start) and
    the secondary x-axis represents 'Time Index'.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to the `PlotGaintable` parent class,
        such as `path_prefix`.
    """

    def __init__(self, **kwargs):
        """
        Initialize the time-based gaintable plotter.

        Sets the primary x-axis label to 'Time (S)' and the secondary x-axis
        label to 'Time Index'.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the `PlotGaintable` parent class.
        """
        super(PlotGaintableTime, self).__init__(**kwargs)
        self._plot_args["x"] = "time"
        self._x_label = "Time (S)"
        self._x_sec_label = "Time Index"

    @property
    def xdim(self):
        """
        str: Specifies the x-dimension name for time plots.

        Returns
        -------
        str
            Returns "time".
        """
        return "time"

    def _prepare_gaintable(self, gain_table, drop_cross_pols=False):
        """
        Prepare the gaintable for time-axis plotting.

        Calls the parent preparation method and then converts the 'time'
        coordinate to seconds elapsed since the first timestep. This relative
        time array is stored in `self._x_data` for axis mapping. It also
        updates the figure title metadata with the observation start time.

        Parameters
        ----------
        gain_table : xarray.Dataset
            The raw gaintable dataset to process.
        drop_cross_pols : bool, optional
            If True, cross-polarization solutions are dropped.
            Defaults to False.

        Returns
        -------
        xarray.Dataset
            The processed gaintable with 'time' converted to relative seconds.
        """
        gaintable = super(PlotGaintableTime, self)._prepare_gaintable(
            gain_table, drop_cross_pols
        )

        starting_time = self.observation_start_time(gaintable)

        self._x_data = gaintable.time - gaintable.time[0]
        self._x_sec_data = np.arange(len(self._x_data))
        self._post_title = f"-[Solution Start Time: {starting_time}]"
        return gaintable.assign({"time": self._x_data})

    def _primary_sec_ax_mapper(self, time_data, time_indexes, reverse=False):
        """
        Create an interpolation function for axis mapping.

        Generates a function to map between relative time (in seconds)
        and the time sample index.

        Parameters
        ----------
        time_data : array-like
            The array of time values (in seconds).
        time_indexes: array-like
            The array of time index values.
        reverse : bool, optional
            If False (default), returns a function that maps
            Time (seconds) -> Time Index.
            If True, returns a function that maps
            Time Index -> Time (seconds).

        Returns
        -------
        callable
            The interpolation function.
        """
        if reverse:
            # Input is time_val (index), map to seconds
            return lambda time_val: np.interp(
                time_val, time_indexes, time_data
            )

        # Input is index (time in seconds), map to index
        return lambda index: np.interp(index, time_data, time_indexes)


class PlotGaintableTargetIonosphere(PlotGaintableFrequency):
    """
    Plots gaintable solutions as a Time vs. Frequency heatmap.

    This class extends `PlotGaintableFrequency` to generate diagnostic plots
    specifically for ionospheric target calibration. It visualizes phase
    variations across both time and frequency channels simultaneously,
    creating a heatmap where the x-axis is Channel/Frequency and the
    y-axis is Time.

    Parameters
    ----------
    path_prefix : str
        A prefix used to construct the output filenames for the plots.
    """

    def __init__(self, path_prefix):
        """
        Initialize the ionosphere target plotter.

        Sets up the specific plotting arguments required for a heatmap
        visualization (e.g., specifying x and y dimensions, enabling
        colorbars).

        Parameters
        ----------
        path_prefix : str
            A prefix used to construct the output filenames for the plots.
        """
        super(PlotGaintableTargetIonosphere, self).__init__(
            path_prefix=path_prefix
        )
        self._plot_args = dict(
            x="Channel",
            y="time",
            col="Station",
            col_wrap=5,
            add_colorbar=True,
            sharex=False,
            aspect=1.5,
        )
        self.__plot_path = f"{path_prefix}-{{plot_type}}-{self.xdim}.png"

    @property
    def xdim(self):
        """
        str: Specifies the dimension identifier for filenames.

        Returns
        -------
        str
            Returns "time-freq" to indicate a dual-domain plot.
        """
        return "time-freq"

    def _prepare_gaintable(self, gain_table, drop_cross_pols=False):
        """
        Prepare the gaintable for Time-Frequency heatmap plotting.

        This method performs the following preprocessing steps:

        1. Calls the parent class (Frequency) preparation.
        2. Converts absolute time to relative time (seconds since start).
        3. Extracts the phase (in degrees) from the complex gains.
        4. Selects only the first Jones element (typically XX or RR) for
           plotting, as ionospheric effects are usually dominated by
           scalar phase terms.

        Parameters
        ----------
        gain_table : xarray.Dataset
            The raw gaintable dataset to process.
        drop_cross_pols : bool, optional
            Ignored in this specific implementation, but kept for signature
            compatibility.

        Returns
        -------
        xarray.Dataset
            The processed gaintable containing a "Phase(Degree)" data
            variable and relative time coordinates.
        """
        gaintable = super(
            PlotGaintableTargetIonosphere, self
        )._prepare_gaintable(gain_table, drop_cross_pols)

        gaintable = gaintable.assign(
            {"time": gaintable.time - gaintable.time[0]}
        )

        gain_phase = gaintable.gain.copy()
        gain_phase.data = np.angle(gain_phase.data, deg=True)

        gaintable = gaintable.assign({"Phase(Degree)": gain_phase})

        # We typically only care about the primary polarization for
        # ionospheric diagnostics.
        return gaintable.isel(Jones_Solutions=[0])

    @dask.delayed
    @safe
    def plot(self, gaintable, figure_title="", **kwargs):
        """
        Generate and save Time vs. Frequency phase heatmaps.

        This is a Dask delayed method that creates a faceted plot where each
        panel represents a station. The color intensity represents the
        phase of the gain solution.

        Parameters
        ----------
        gaintable : xarray.Dataset
            The input gaintable dataset to plot.
        figure_title : str, optional
            A prefix for the main figure title. Defaults to "".
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        None
        """
        y_label = "Time (S)"
        starting_time = self.observation_start_time(gaintable)

        gaintable = self._prepare_gaintable(gaintable)
        facet_plot = gaintable["Phase(Degree)"].plot(**self._plot_args)

        facet_plot.set_axis_labels(self._x_label, y_label)
        for ax in facet_plot.axs.reshape(-1):
            ax.tick_params(labelbottom=True)

        self._update_facet(facet_plot, y_label)
        facet_plot.set_ticks(fontsize=8)

        gain_phase_fig = facet_plot.fig

        gain_phase_fig.suptitle(
            (
                f"{figure_title} Solutions (Phase)-"
                f"[Solution Start Time: {starting_time}]"
            ),
            fontsize="x-large",
            y=1.08,
        )
        gain_phase_fig.tight_layout()
        # Adjust right margin to make room for the colorbar
        plt.subplots_adjust(right=0.83)
        gain_phase_fig.savefig(
            self.__plot_path.format(plot_type="phase"), bbox_inches="tight"
        )

        plt.close()
