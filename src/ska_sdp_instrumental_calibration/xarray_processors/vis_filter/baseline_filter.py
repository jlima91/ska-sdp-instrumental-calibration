import re
from typing import Dict

import numpy as np
import xarray as xr

from ...data_managers.baseline_expression import BaselinesExpression
from .._utils import parse_antenna
from .vis_filter import AbstractVisibilityFilter


class BaselineFilter(AbstractVisibilityFilter):
    """
    Filters xarray datasets based on string-based baseline specifications.

    Parses a comma-separated string of baseline expressions (e.g., "1~5 & 2")
    and generates boolean masks to flag or retain data.

    Parameters
    ----------
    baselines : str
        A comma-separated string of baseline expressions. Each expression
        defines a set of antennas or ranges to select (e.g., "1~5 & 6~8").
    station_names : xr.DataArray
        DataArray containing the names of stations/antennas.
    station_counts : int
        The total count of stations available.

    Attributes
    ----------
    _baseline_pattern : str
        Regex pattern to identify negation, left, and right operands.
    """

    _FILTER_NAME_ = "baselines"

    _baseline_pattern = (
        r"^(?P<negate>!?)(?P<left>[a-zA-Z0-9~*]+)"
        r"(?:\s*&\s*(?P<right>[a-zA-Z0-9~*]+))?$"
    )
    _re_baseline = re.compile(_baseline_pattern)

    @classmethod
    def filter(cls, baselines: str, vis: xr.Dataset):
        baseline_filter = BaselineFilter(baselines, vis.configuration.names)

        return baseline_filter(
            vis.baselines,
            vis.flags,
        )

    def __init__(self, baselines: str, station_names: xr.DataArray):
        self.__baseline_expressions = []
        if baselines:
            self.__baseline_expressions = [
                BaselinesExpression(
                    **self.__parse(baseline_expr),
                    antenna_parser=lambda ant: parse_antenna(
                        ant, station_names
                    ),
                )
                for baseline_expr in baselines.split(",")
            ]

    def __parse(self, baseline_expr: str) -> Dict[str, str]:
        """
        Parse a single baseline expression string into its components.

        Uses regex to extract the negation flag, left-hand side, and
        optional right-hand side of the expression.

        Parameters
        ----------
        baseline_expr : str
            The raw string expression (e.g., "!1~3 & 5").

        Returns
        -------
        dict
            A dictionary containing keys 'negate', 'left', and 'right'.

        Raises
        ------
        ValueError
            If the expression does not match the expected regex pattern.
        """
        match = self._re_baseline.match(baseline_expr.strip())
        if match:
            return match.groupdict()

        raise ValueError(f"Invalid baseline expression: {baseline_expr}")

    def __call__(
        self, baselines: xr.DataArray, flags: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply the filters to the provided baselines and update flags.

        Evaluates all stored baseline filters. If a baseline matches *any*
        of the filters, it is considered 'selected' (valid). Baselines that
        do not match any filter are flagged (set to True).

        Parameters
        ----------
        baselines : xr.DataArray
            The array of baseline coordinates/indices to evaluate.
        flags : xr.DataArray
            The existing flag array to update.

        Returns
        -------
        xr.DataArray
            The updated flag array. Baselines not matching the selection
            criteria are marked as flagged (True).
        """

        masks = [
            baseline_filter.predicate(baselines.data)
            for baseline_filter in self.__baseline_expressions
        ]

        if not masks:
            return flags

        is_selected = xr.ones_like(baselines, dtype=bool).where(
            np.logical_or.reduce(masks), other=False
        )

        new_flags = xr.zeros_like(flags, dtype=bool).where(
            is_selected, other=True
        )

        return flags | new_flags
