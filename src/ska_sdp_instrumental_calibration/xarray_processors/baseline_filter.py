import re

import xarray as xr

from ._utils import parse_antenna


class BaselineFilter:
    """
    A filter for selecting or ignoring specific baselines in
    interferometric data.

    The BaselineFilter class parses a user-specified, comma-separated string of
    baselines (e.g., 'ANT1&ANT2,3&ANT4') and constructs a filter to ignore
    those baselines during data processing. Baselines can be specified by name
    or index. The filter can then be applied to xarray DataArrays representing
    baselines and their associated flags, updating the flags to ignore the
    specified baselines.

    A comma-separated string defining baselines to ignore
    (e.g., 'ANT1&ANT2,3&ANT4').

    """

    _baseline_pattern = r"(\w+)\s*&\s*(\w+)"
    _re_baseline = re.compile(_baseline_pattern)

    def __init__(
        self, baselines: str, station_names: xr.DataArray, station_counts: int
    ):
        """
        Initialize the BaselineFilter with a comma-separated list of baselines.

        Parameters
        ----------
        baselines : str
            A comma-separated string defining baselines (e.g.,
            'ANT1&ANT2,3&ANT4'). If empty, no filtering is applied.

        station_names : xr.DataArray
            An xarray DataArray containing the names of the stations/antennas.

        station_counts : int
            The total number of stations/antennas.

        Raises
        ------
        ValueError
            If a baseline string cannot be parsed.
        """
        matches = []
        if baselines:
            matches = re.findall(self._re_baseline, baselines)
            if not matches:
                raise ValueError(
                    f"Could not parse baselines from '{baselines}'"
                )
        self._baselines_to_ignore = [
            self.__parse_baseline(ant1, ant2, station_names, station_counts)
            for ant1, ant2 in (matches)
        ]

    def __parse_baseline(
        self,
        antenna1,
        antenna2,
        station_names: xr.DataArray,
        station_counts: int,
    ):
        try:
            antenna1 = int(antenna1)
        except ValueError:
            pass

        try:
            antenna2 = int(antenna2)
        except ValueError:
            pass

        return parse_antenna(
            antenna1, station_names, station_counts
        ), parse_antenna(antenna2, station_names, station_counts)

    def __call__(self, baselines: xr.DataArray, flags: xr.DataArray):
        """
        Initialize the BaselineFilter with a comma-separated list of baselines.

        Parameters
        ----------
        baselines : xr.DataArray
            A DataArray defining baselines (e.g.,
            'ANT1&ANT2,3&ANT4'). If empty, no filtering is applied.

        flags : xr.DataArray
            A DataArray defining the flags for the baselines.
        """
        if not self._baselines_to_ignore:
            return flags

        baseline_filter = xr.ones_like(baselines, dtype=bool).where(
            [
                baseline not in self._baselines_to_ignore
                for baseline in baselines.data
            ],
            other=False,
        )

        # import pdb; pdb.set_trace()
        return flags | xr.zeros_like(flags, dtype=bool).where(
            baseline_filter, other=True
        )
