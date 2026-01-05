import re

import numpy as np
import xarray as xr
from astropy import constants as const

from ...data_managers.uv_range import UVRange
from .vis_filter import VisibilityFilter


class UVRangeFilter(VisibilityFilter):
    """
    Parses and applies CASA-style UV range selection strings to UVW coordinates

    This class handles parsing of UV range strings (e.g., '0~10klambda',
    '>500m') and generates boolean masks for data filtering based on calculated
    UV distances.

    Attributes
    ----------
    _num_re : str
        Regex pattern for matching floating point numbers.
    _re_range : re.Pattern
        Regex pattern for matching range strings (e.g., "0~10klambda").
    _re_ineq : re.Pattern
        Regex pattern for matching inequality strings (e.g., ">500m").
    """

    _FILTER_NAME_ = "uvdist"

    _num_re = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    # Matches: "!0~10klambda"
    _re_range = re.compile(
        rf"^(!?)\s*({_num_re})\s*~\s*({_num_re})\s*([a-z]*)$"
    )

    # Matches: ">500m"
    _re_ineq = re.compile(rf"^(!?)\s*(>|<)\s*({_num_re})\s*([a-z]*)$")

    @classmethod
    def _filter(cls, uvdist: str, vis: xr.Dataset):
        uvrange_filter = UVRangeFilter(uvdist)

        return uvrange_filter(
            vis.uvw.sel(spatial="u", drop=True),
            vis.uvw.sel(spatial="v", drop=True),
            vis.flags,
            vis.frequency,
        )

    def __init__(self, uvranges):
        """
        Initialize the UVRangeFilter with a comma-separated list of ranges.

        Parameters
        ----------
        uvranges : str
            A comma-separated string defining UV ranges (e.g.,
            '0~10klambda,>500m'). Supports units 'm', 'km', 'klambda' (or
            'kl'). If empty, no filtering is applied.

        Raises
        ------
        ValueError
            If a provided unit is unknown or if a range string cannot be parsed
        """
        self.__needs_klambda = False
        self._uvranges = (
            [
                UVRange(**self.__parse(uvrange))
                for uvrange in uvranges.split(",")
            ]
            if uvranges
            else []
        )

    def __parse_unit(self, unit):
        """
        Normalize the unit string and determine if wavelength conversion is
        needed.

        Parameters
        ----------
        unit : str
            The unit string to parse (e.g., 'm', 'klambda', 'km').

        Returns
        -------
        str
            The normalized unit ('m', 'km', or 'kl').

        Raises
        ------
        ValueError
            If the unit string is not recognized.
        """
        if unit and unit not in [
            "m",
            "meter",
            "meters",
            "km",
            "kilometer",
            "kl",
            "klambda",
            "kilolambda",
        ]:
            raise ValueError(f"Unknown unit: {unit}")

        if unit in ("klambda", "kilolambda", "kl"):
            self.__needs_klambda = True
            return "kl"

        if unit in ("kilometer", "km"):
            return "km"

        return "m"

    def __parse(self, uvrange):
        """
        Parse a single CASA-style uvrange string into parameter dictionary.

        Supported Syntax:
          - Ranges: '0~10klambda', '100~500m'
          - Inequalities: '>500m', '<10kl'
          - Inversion: '!>500m' (Select everything NOT > 500m)
          - Units: 'm' (default), 'km', 'klambda' (or 'kl')

        Parameters
        ----------
        uvrange : str
            The raw UV range string to parse.

        Returns
        -------
        dict
            A dictionary containing parameters suitable for initializing a
            ``UVRange`` object (e.g., uv_min, uv_max, negate, unit).

        Raises
        ------
        ValueError
            If the ``uvrange`` string format matches neither range nor
            inequality patterns.
        """
        # Clean input
        selection_str = uvrange.strip().lower()

        params = {
            "uv_min": -np.inf,
            "uv_max": np.inf,
            "min_inclusive": True,
            "max_inclusive": True,
        }

        operator_key = {">": "uv_min", "<": "uv_max"}

        operator_inclusive = {
            ">": {"min_inclusive": False},
            "<": {"max_inclusive": False},
        }

        match_range = self._re_range.match(selection_str)
        match_ineq = self._re_ineq.match(selection_str)

        if match_range:
            negate, start, end, unit = match_range.groups()
            return {
                **params,
                "negate": negate == "!",
                "uv_min": float(start),
                "uv_max": float(end),
                "unit": self.__parse_unit(unit),
            }

        if match_ineq:
            negate, op, val, unit = match_ineq.groups()
            return {
                **params,
                **operator_inclusive[op],
                "negate": negate == "!",
                operator_key[op]: float(val),
                "unit": self.__parse_unit(unit),
            }

        raise ValueError(f"Could not parse uvrange string: '{selection_str}'")

    def __call__(
        self, u: xr.DataArray, v: xr.DataArray, flags: xr.DataArray, freq=None
    ):
        """
        Apply the UV range filter to update flag arrays.

        Calculates UV distances and determines which data points fall within
        the specified ranges. Flags are updated where data falls *outside* the
        selected ranges (unless negated).

        Parameters
        ----------
        u : xr.DataArray
            U coordinates in meters.
        v : xr.DataArray
            V coordinates in meters.
        flags : xr.DataArray
            The existing boolean flag array to update.
        freq : float or array_like, optional
            Frequency in Hz. Required if any range uses 'klambda' units.

        Returns
        -------
        xarray.DataArray
            Updated boolean flag array where 1 (True) indicates flagged data.
            If no ranges were provided during initialization, the original
            flags are returned unchanged.

        Raises
        ------
        ValueError
            If 'klambda' units are used but ``freq`` is not provided.
        """
        uvdist = np.hypot(u, v)
        uvdist_kl = None
        if self.__needs_klambda:
            if freq is None:
                raise ValueError("Frequency required for 'klambda' selection.")
            _lambda = const.c.value / freq  # pylint: disable=no-member
            uvdist_kl = (uvdist / _lambda) / 1000.0

        masks = [
            uvrange.predicate(uvdist, uvdist_kl) for uvrange in self._uvranges
        ]

        if len(masks) == 0:
            return flags

        is_selected = masks[0]
        # Looping for dask compatibility
        for msk in masks[1:]:
            is_selected = is_selected | msk

        new_flags = xr.zeros_like(flags, dtype=bool).where(
            is_selected, other=True
        )

        return flags | new_flags
