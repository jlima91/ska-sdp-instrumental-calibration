import numpy as np


def channel_frequency_mapper(frequency, reverse=False):

    if reverse:
        return lambda freq: np.interp(
            freq, frequency, np.arange(len(frequency))
        )

    return lambda channel: np.interp(
        channel, np.arange(len(frequency)), frequency
    )


class XDim:
    """
    Parent class for choosing the X dimension in plot
    """

    pass


class XDim_Frequency(XDim):
    """
    Configuration and helper methods for handling the 'frequency' dimension.

    This class provides static methods to format a Matplotlib axis
    and extract relevant data when the primary x-dimension is 'frequency'.

    Attributes
    ----------
    label : str
        The name of the dimension, set to "frequency".

    """

    label = "freq"

    @staticmethod
    def x_axis(primary_ax, secondary_ax, frequency):
        """
        Sets x-axis to display frequency in MHz.

        This method assumes the primary x-axis represents channels and
        creates a secondary axis at the top to show the corresponding
        frequency in MHz.

        Parameters
        ----------
        primary_ax : matplotlib.axes.Axes
            The Matplotlib axes object to be modified.
        secondary_ax : matplotlib.axes.Axes
            The Matplotlib axes object to be modified.
        gaintable : xarray.DataSet
            Gaintable

        Returns
        -------
        None
            Modifies the `axis` object in-place.

        """
        primary_ax.set_xlabel("Channel")

        secondary_ax.secondary_xaxis(
            "top",
            functions=(
                channel_frequency_mapper(frequency),
                channel_frequency_mapper(frequency, reverse=True),
            ),
        ).set_xlabel("Frequency [MHz]")

    @staticmethod
    def data(gaintable):
        """
        Extracts the frequency data from a gaintable and converts to MHz.

        Parameters
        ----------
        gaintable : xarray.DataSet
            Gaintable

        Returns
        -------
        numpy.ndarray or array_like
            The frequency data array converted from Hz to MHz (divided by 1e6).

        """
        frequency = gaintable.frequency / 1e6
        return (np.arange(len(frequency)), frequency)

    @staticmethod
    def gain(gaintable, antenna):
        """
        Selects the gain data for along the frequency dimension

        Parameters
        ----------
        gaintable: xarray.DataSet
            Gaintable
        antenna: int
            Antenna index
        """
        return gaintable.gain.isel(time=0, antenna=antenna)


class XDim_Time(XDim):
    """
    Utility class for handling the 'time' dimension.

    Provides static methods for plotting configurations (e.g., axis labels)
    and data extraction/scaling related to time.

    Attributes
    ----------
    label : str
        The name of the dimension, hardcoded to "time".
    """

    label = "time"

    @staticmethod
    def x_axis(axis, *args):
        """
        Sets  X-axis label for time.

        Modifies the provided matplotlib axis object in-place.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            The axis object to modify.
        *args
            Variable length arguments (ignored by this method).
        """
        axis.set_xlabel("Time (S)")

    @staticmethod
    def data(gaintable):
        """
        Extracts and scales time data from a gaintable.

        Assumes the 'time' attribute in the gaintable is stored
        in nanoseconds and converts it to seconds.

        Parameters
        ----------
        gaintable : xarray.DataSet
            Gaintable
        Returns
        -------
        numpy.ndarray or float
            The time data converted to seconds (time / 1e9).
        """
        return gaintable.time, None

    @staticmethod
    def gain(gaintable, antenna):
        """
        Selects the gain data for along the time dimension

        Parameters
        ----------
        gaintable: xarray.DataSet
            Gaintable
        antenna: int
            Antenna index
        """
        return gaintable.gain.isel(frequency=0, antenna=antenna)
