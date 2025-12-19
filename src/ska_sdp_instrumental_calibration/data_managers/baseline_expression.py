import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Union

import numpy as np


@dataclass(slots=True)
class BaselinesExpression:
    """
    Parses and evaluates baseline selection expressions.

    This class handles logic for selecting baselines based on string
    expressions representing antenna ranges.

    Parameters
    ----------
    left : str
        The left-hand side of the baseline expression (e.g., "1" or "1~5").
    right : str
        The right-hand side of the baseline expression.
    negate : Union[bool, str]
        Flag to indicate if the selection should be inverted. If passed as
        "!", it is converted to True (exclusion).
    antenna_parser : Callable, optional
        Function to parse antenna strings into integers. Defaults to
        ``lambda x: int(x)``.
    """

    left: str
    right: str
    negate: Union[bool, str]
    antenna_parser: Callable = lambda x: int(x)

    def __post_init__(self):
        """
        Normalize the negation flag to a boolean after initialization.
        """
        if isinstance(self.negate, str):
            self.negate = self.negate == "!"

        self.left = self.__parse_range(self.left)
        self.right = self.__parse_range(self.right)

    def __parse_range(self, expression: str) -> range:
        """
        Parse a string expression into a Python range object.

        Parameters
        ----------
        expression : str
            The string defining the range (e.g., "5" or "1~4").

        Returns
        -------
        range
            A range object covering the specified start and end (inclusive).
        """
        start, *end = [self.antenna_parser(x) for x in expression.split("~")]
        end = end[0] if len(end) else start
        return range(start, end + 1)

    def predicate(self, baselines: Iterable[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate a boolean mask for the provided baselines.

        Determines which of the input baselines match the left and right
        antenna criteria defined in this expression.

        Parameters
        ----------
        baselines : iterable of tuples
            A collection of baselines (tuples of antenna IDs) to evaluate.

        Returns
        -------
        np.ndarray
            A boolean array of the same length as ``baselines``, where True
            indicates the baseline matches the expression.
        """

        valid_set = set(itertools.product(self.left, self.right))

        mask = np.array([b in valid_set for b in baselines], dtype=bool)

        if self.negate:
            return ~mask

        return mask
