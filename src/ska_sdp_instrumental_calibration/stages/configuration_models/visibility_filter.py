from typing import Annotated, Optional

from pydantic import Field
from ska_sdp_piper.piper import PiperBaseModel


class VisibilityFilterConfig(PiperBaseModel):
    """
    A model describing the Visibility filter config passed
    to the Bandpass Calibration stage
    """

    uvdist: Annotated[
        Optional[str],
        Field(
            description="""
            CASA like strings which determine
            which uv ranges to keep in the filtered data.
            Separated by comma for multiple.
            Default unit is set to be meter.
            For e.g. '0~10klambda' ; '>10m,<100m'
            """
        ),
    ] = None
    exclude_baselines: Annotated[
        Optional[str],
        Field(
            description="""
            CASA like strings which determine which baselines to exclude
            in the filtered data.
            A baseline is formed using antenna indices or antenna names,
            where a pair is joined using '&'.
            Each baseline must be seperated by comma.
            For e.g. 'ANT1&ANT2,1~3&ANT4' will exclude following
            baselines: ant1&ant2, ant1&ant4, ant2&ant4, ant3&ant4
            """
        ),
    ] = None
