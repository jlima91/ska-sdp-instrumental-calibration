import operator
from dataclasses import dataclass


@dataclass(slots=True)
class UVRange:
    """
    Data container for a single UV range selection rule.

    This class encapsulates the parameters defining a specific UV range,
    including boundaries, units, inclusivity, and negation logic.
    It is typically instantiated by ``UVRangeFilter`` rather than directly
    by the user.
    """

    uv_min: float
    "Minimum UV distance for the range."
    uv_max: float
    "Maximum UV distance for the range."
    unit: str
    """The unit of the range. Must be one of 'm', 'km', or 'kl'."""
    negate: bool
    """If True, the selection logic is inverted (i.e., select data *outside*
    the specified range)."""
    min_inclusive: bool
    "If True, use inclusive comparison (>=) for the lower bound."
    max_inclusive: bool
    "If True, use inclusive comparison (<=) for the upper bound."

    def __post_init__(self):
        """
        Validate the range parameters after initialization.

        Raises
        ------
        ValueError
            If ``uv_min`` is greater than ``uv_max``.
        """
        if self.uv_min > self.uv_max:
            raise ValueError(
                f"Invalid range: min > max ({self.uv_min} > {self.uv_max})"
            )

    def __convert(self, uvdist, uvdist_kl):
        """
        Convert input UV distances to the unit specified by this range.

        Parameters
        ----------
        uvdist : array_like
            UV distance in meters.
        uvdist_kl : array_like or None
            UV distance in kilolambda.

        Returns
        -------
        array_like
            The UV distance converted to the unit stored in ``self.unit``.

        Raises
        ------
        ValueError
            If ``self.unit`` is not one of 'm', 'km', or 'kl'.
        """
        if self.unit == "m":
            return uvdist

        if self.unit == "km":
            return uvdist / 1000.0

        if self.unit == "kl":
            return uvdist_kl

        raise ValueError(f"Unknown unit: {self.unit}")

    def predicate(self, uvdist, uvdist_kl):
        """
        Evaluate the range condition against provided UV distances.

        Applies the lower and upper bound checks, inclusivity rules, and
        negation logic to determine which points fall within the range.

        Parameters
        ----------
        uvdist : array_like
            UV distance in meters.
        uvdist_kl : array_like or None
            UV distance in kilolambda. Required if ``self.unit`` is 'kl'.

        Returns
        -------
        array_like
            Boolean mask where True indicates the data satisfies the range
            criteria.
        """
        min_comparison = operator.ge if self.min_inclusive else operator.gt
        max_comparison = operator.le if self.max_inclusive else operator.lt

        uvdist = self.__convert(uvdist, uvdist_kl)
        cond = min_comparison(uvdist, self.uv_min) & max_comparison(
            uvdist, self.uv_max
        )
        if self.negate:
            return ~cond

        return cond
