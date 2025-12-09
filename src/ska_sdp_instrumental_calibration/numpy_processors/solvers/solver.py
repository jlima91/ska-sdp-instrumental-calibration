from typing import Tuple

import numpy as np
import xarray as xr

from .solvers_factory import SolverFactory


class Solver(metaclass=SolverFactory):
    """
    Base class for gain solvers.

    This class provides a common interface and shared configuration for
    calibration solvers. Subclasses should implement the `solve` method
    for specific algorithms.

    Parameters
    ----------
    niter : int, optional
        Maximum number of iterations for the solver. Default is 50.
    tol : float, optional
        Convergence tolerance for the iterative solver. Default is 1e-6.
    **_ : dict
        Additional keyword arguments (ignored by the base class).

    Attributes
    ----------
    niter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    norm_method : str or None
        Method used for gain normalization. Default is None.
    """

    def __init__(self, niter=50, tol=1e-6, **_):
        self.niter = niter
        self.tol = tol
        self.norm_method = None

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
        Run the calibration solver (abstract method).

        This method defines the standard signature for all solver
        implementations. It accepts observed data, model data, and initial
        gain estimates, returning updated gains.

        Parameters
        ----------
        vis_vis : np.ndarray
            Observed visibility data. Shape: (time, baseline, freq, pol).
        vis_flags : np.ndarray
            Flags for observed visibilities. Shape matches `vis_vis`.
        vis_weight : np.ndarray
            Weights for observed visibilities. Shape matches `vis_vis`.
        model_vis : np.ndarray
            Model visibility data. Shape matches `vis_vis`.
        model_flags : np.ndarray
            Flags for model visibilities. Shape matches `vis_vis`.
        gain_gain : np.ndarray
            Initial gain estimates. Shape: (time, ant, freq, rec1, rec2).
        gain_weight : np.ndarray
            Storage for gain weights. Shape matches `gain_gain`.
        gain_residual : np.ndarray
            Storage for gain residuals. Shape matches `gain_gain`.
        ant1 : np.ndarray
            Indices of antenna 1 for each baseline. Shape: (nbl,).
        ant2 : np.ndarray
            Indices of antenna 2 for each baseline. Shape: (nbl,).

        Returns
        -------
        tuple of np.ndarray
            A tuple containing:

            - Updated gain array.
            - Gain weights.
            - Gain residuals.

        Raises
        ------
        NotImplementedError
            This method must be overridden by subclasses.
        """
        raise NotImplementedError("solve not implemented")

    def normalise_gains(self, gain: xr.DataArray) -> xr.DataArray:
        """
        Normalize gain amplitudes.

        This utility allows solvers to apply constraints (like unit amplitude)
        or normalization conventions (like mean amplitude) to the solutions.

        Parameters
        ----------
        gain : xarray.DataArray
            The gain table to normalize.

        Returns
        -------
        xarray.DataArray
            The normalized gain table.

        Raises
        ------
        NotImplementedError
            If `norm_method` is set to a value that is not implemented
            by the base class.
        """
        if self.norm_method is None:
            return gain

        raise NotImplementedError(
            f"Normalise gains using {self.norm_method} not implemented"
        )
