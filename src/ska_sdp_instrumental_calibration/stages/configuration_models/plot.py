from typing import Annotated

from pydantic import Field
from ska_sdp_piper.piper import PiperBaseModel


class PlotConfig(PiperBaseModel):
    plot_table: Annotated[
        bool,
        Field(description="Plot the generated gaintable"),
    ] = True
    fixed_axis: Annotated[
        bool,
        Field(description="Limit amplitude axis to [0-1]"),
    ] = False


class PlotRMConfig(PiperBaseModel):
    """
    A model describing the RM Plot config passed
    to the Generate Channel RM stage
    """

    plot_rm: Annotated[
        bool,
        Field(
            description="""Plot the estimated rotational measures
            per station"""
        ),
    ] = False
    station: Annotated[
        int | str,
        Field(description="Station number/name to be plotted"),
    ] = 0


class PlotFlagGainConfig(PiperBaseModel):
    """
    A model describing the Plot config passed
    to the Flag Gain stage
    """

    curve_fit_plot: Annotated[
        bool,
        Field(description="Plot the fitted curve of gain flagging"),
    ] = True
    gain_flag_plot: Annotated[
        bool,
        Field(description="Plot the flagged weights"),
    ] = True


class PlotSmoothGainsConfig(PiperBaseModel):
    """
    A model describing the Plot Config config passed
    to the Smooth Gain Solution stage
    """

    plot_table: Annotated[
        bool,
        Field(
            description="""Plot the smoothed gaintable""",
        ),
    ] = False
    plot_path_prefix: Annotated[
        str,
        Field(
            description="""Path prefix to store smoothed gain plots""",
        ),
    ] = "smoothed-gain"
    plot_title: Annotated[
        str,
        Field(
            description="""Title for smoothed gain plots""",
        ),
    ] = "Smoothed Gain"
