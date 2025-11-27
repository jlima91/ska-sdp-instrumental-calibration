from typing import Literal, Union

import numpy as np
import xarray as xr


class SolutionIntervals:
    """
    A group of intervals over time data of visibility.
    The INST pipeline performs calibration of
    each interval individually.

    Parameters
    ----------
    time: np.ndarray
        The numpy array containing time values of visibility
    timeslice: float or "full", optional
        Determines the width of each solution interval
    """

    def __init__(
        self,
        time: np.ndarray,
        timeslice: Union[float, Literal["full"], None] = None,
    ):
        dim_name = "time"
        time_xdr = xr.DataArray(time, coords={dim_name: time})

        if timeslice == "full":
            nbins = 1
        elif (timeslice is None) or (timeslice <= 0.0):
            nbins = time_xdr.size
        else:
            # Determine number of equal width bins
            # TODO: Should bins always be of equal interval?
            nbins = min(
                max(
                    1,
                    int(
                        np.ceil(
                            (
                                time_xdr[dim_name].data.max()
                                - time_xdr[dim_name].data.min()
                            )
                            / timeslice
                        )
                    ),
                ),
                time_xdr[dim_name].size,
            )

        self._time_bins = time_xdr.groupby_bins(dim_name, nbins, squeeze=False)

    @property
    def size(self):
        """
        Returns number of solution intervals

        Returns
        -------
        int
        """
        return len(self._time_bins)

    @property
    def solution_time(self):
        """
        Returns time values for each solution interval

        Returns
        -------
        np.ndarray
        """
        return self._time_bins.mean().data

    @property
    def indices(self):
        """
        Returns list of slice objects corresponding
        to indices of each solution interval
        If converting to slice is not possible, it
        returns the indices

        Returns
        -------
        list[slice | list[int]]
        """
        return [
            self.__indices_to_slice(idx)
            for idx in self._time_bins.groups.values()
        ]

    @property
    def intervals(self):
        """
        Returns numpy array containing the
        width of each solution interval

        Returns
        -------
        np.ndarray
        """
        return np.array(
            [(iv.right - iv.left) for iv in self._time_bins.groups.keys()]
        )

    def __indices_to_slice(
        self,
        index_array: np.ndarray | list[int],
    ) -> slice | np.ndarray:
        """
        Convert a 1D integer index array to a slice **if** it represents
        an arithmetic progression: same step between all consecutive elements.
        Otherwise return the original index array.

        Parameters
        ----------
        index_array : array_like of int

        Returns
        -------
        slice or numpy array
        """
        index_array = np.asarray(index_array)

        # Empty input
        if index_array.size == 0:
            return index_array

        # Must be integer-valued
        if not np.issubdtype(index_array.dtype, np.integer):
            return index_array

        # One element → trivial slice
        if index_array.size == 1:
            i = index_array[0]
            return slice(i, i + 1, 1)

        # Sort (groupby_bins etc. may produce unordered indices)
        sorted_idx = np.sort(index_array)

        # Compute steps between elements
        steps = np.diff(sorted_idx)

        # Step must be consistent
        step = steps[0]
        if np.all(steps == step):
            start = sorted_idx[0]
            stop = sorted_idx[-1] + step
            return slice(start, stop, step)

        # Not a simple arithmetic progression → return indices
        return index_array
