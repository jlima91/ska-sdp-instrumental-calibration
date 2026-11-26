"""
Cable delays of a field: the median X and Y delay of each station, over
the times of its delay table.
"""

import xarray as xr

from ..domain.cable_delays import CableDelays


def get_cable_delays(delaytable: xr.Dataset) -> CableDelays:
    """
    Get the cable delays of a field from its delay table.

    A cable delay does not change over an observation, so each
    station gets the median of its delays over time: unlike the mean,
    it is barely moved by the delays of one badly solved scan. Delays
    that were not solved (NaN) are left out.

    Parameters
    ----------
    delaytable : DelayTable
        Delay table of the field, with ``delay`` in seconds of
        dimensions (time, antenna_name, pol) and ``XX`` and ``YY``
        polarisations.

    Returns
    -------
    CableDelays
        Cable delays of the field.
    """
    # Nanoseconds, as in CABLE_DELAYS_COLUMNS
    median_delays = delaytable.delay.median(dim="time", skipna=True) * 1e9
    x_delays = median_delays.sel(pol="XX").values
    y_delays = median_delays.sel(pol="YY").values

    return CableDelays(
        {
            str(name): {"x_delay": float(x), "y_delay": float(y)}
            for name, x, y in zip(
                delaytable.antenna_name.values, x_delays, y_delays
            )
        }
    )
