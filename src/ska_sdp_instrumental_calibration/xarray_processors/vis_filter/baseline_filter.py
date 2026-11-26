import re
from typing import Dict

import numpy as np
import xarray as xr

from ...data_managers.baseline_expression import BaselinesExpression
from ...data_managers.processing_set import (
    get_baseline_antenna_indices,
    get_data_group,
)
from .._utils import parse_antenna
from .vis_filter import VisibilityFilter


class BaselineFilter(VisibilityFilter):
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

    _FILTER_NAME_ = "exclude_baselines"

    _baseline_pattern = (
        r"^(?P<left>[a-zA-Z0-9~*]+)" r"(?:\s*&\s*(?P<right>[a-zA-Z0-9~*]+))?$"
    )
    _re_baseline = re.compile(_baseline_pattern)

    @classmethod
    def _filter(cls, baselines: str, vis: xr.DataTree):
        baseline_filter = BaselineFilter(baselines, _station_names(vis))

        return baseline_filter(
            _baseline_antenna_pairs(vis),
            vis.ds[get_data_group(vis)["flag"]],
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
            The raw string expression (e.g., "1~3 & 5").

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
            np.logical_and.reduce(masks), other=False
        )

        new_flags = xr.zeros_like(flags, dtype=bool).where(
            is_selected, other=True
        )

        return flags | new_flags


def _station_names(vis: xr.DataTree) -> xr.DataArray:
    """
    Antenna names of an MSv4 node, in ``antenna_xds`` order, with an ``id``
    coordinate holding their index.

    Replaces ``Visibility.configuration.names``, as ``parse_antenna``
    resolves antenna names through that ``id`` coordinate.
    """
    names = vis["antenna_xds"].ds.antenna_name.values
    return xr.DataArray(names, dims="id", coords={"id": np.arange(names.size)})


def _baseline_antenna_pairs(vis: xr.DataTree) -> xr.DataArray:
    """
    (antenna1, antenna2) index pairs of the baselines of an MSv4 node, in
    ``antenna_xds`` order and in the antenna order of the data.

    Replaces ``Visibility.baselines``, whose values are such pairs.
    """
    antenna1, antenna2 = get_baseline_antenna_indices(vis)
    pairs = np.empty(antenna1.size, dtype=object)
    pairs[:] = list(zip(antenna1.tolist(), antenna2.tolist()))
    return xr.DataArray(
        pairs,
        dims="baseline_id",
        coords={"baseline_id": vis.ds.baseline_id.values},
    )
