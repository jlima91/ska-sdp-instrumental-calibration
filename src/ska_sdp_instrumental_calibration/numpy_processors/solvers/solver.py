from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import xarray as xr


class Solver(ABC):
    """
    Base class for gain solvers.

    This class provides a common interface and shared configuration for
    calibration solvers. Subclasses should implement the `solve` method
    for specific algorithms.

    Parameters
    ----------
    niter : int, optional
        Maximum number of iterations for the solver (default is 30).
    tol : float, optional
        Convergence tolerance for the iterative solver (default is 1e-6).

    Notes
    -----
    This class is intended to be subclassed for specific solver
    implementations, such as gain substitution or Jones matrix solvers.
    The `solve` method should be overridden in derived classes.
    """

    def __init__(self, niter=50, tol=1e-6, **_):
        self.niter = niter
        self.tol = tol

    @abstractmethod
    def solve(
        self,
        vis_vis: np.ndarray,
        vis_flags: np.ndarray,
        vis_weight: np.ndarray,
        model_vis: np.ndarray,
        model_flags: np.ndarray,
        gain_gain: np.ndarray,
        gain_weight: np.ndarray,
        gain_residual: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Abstract solve method.

        Parameters
        ----------
        vis_vis : numpy.ndarray
            Observed visibility data array, shape [time, baseline, freq, pol].
        vis_flags : numpy.ndarray
            Flags for observed visibilities, same shape as vis_vis.
        vis_weight : numpy.ndarray
            Weights for observed visibilities, same shape as vis_vis.
        model_vis : numpy.ndarray
            Model visibility data array, shape vis_vis.
        model_flags : numpy.ndarray
            Flags for model visibilities, same shape as model_vis.
        gain_gain : numpy.ndarray
            Array to store the estimated gains,
            shape [time, ant, freq, rec1, rec2]
        gain_weight : numpy.ndarray
            Array to store the gain weights, same shape as gain_gain.
        gain_residual : numpy.ndarray
            Array to store the gain residuals, same shape as gain_gain.
        ant1 : numpy.ndarray
            Indices of the first antenna for each baseline.
        ant2 : numpy.ndarray
            Indices of the second antenna for each baseline.

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Updated gain, weights, and residual arrays after calibration.
        """
        pass

    def normalise_gains(self, gain: xr.DataArray) -> xr.DataArray:
        """
        Function to normalize gains

        Parameters
        ----------
        gaintable: xarray.DataArray

        Returns
        -------
        xarray.DataArray
        """
        return gain
