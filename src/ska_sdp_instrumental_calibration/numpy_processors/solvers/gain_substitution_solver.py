import logging
from typing import Tuple

import numpy as np
import xarray as xr

from .processing_functions import create_point_vis, gain_substitution
from .solver import Solver

logger = logging.getLogger(__name__)


class GainSubstitution(Solver):
    """
    Solver for antenna gains using the gain substitution algorithm.

    This class implements the iterative gain substitution method for solving
    antenna-based complex gains in radio interferometric calibration. It
    compares observed visibilities against a model to derive gain corrections.

    Parameters
    ----------
    refant : int, optional
        Index of the reference antenna. The phase of this antenna is clamped
        to zero during the solution. Default is 0.
    phase_only : bool, optional
        If True, solve only for the phase of the gains, keeping amplitudes
        constant. Default is False.
    crosspol : bool, optional
        If True, solve for cross-polarization terms (e.g., XY, YX, RL, LR)
        in addition to parallel hands. Default is False.
    normalise_gains : str, optional
        Method to normalize gain amplitudes. Options are 'mean', 'median',
        or None. Default is None.
    **kwargs
        Additional keyword arguments passed to the base `Solver` class
        (e.g., `niter`, `tol`).

    Attributes
    ----------
    refant : int
        Reference antenna index.
    phase_only : bool
        Whether to solve for phase only.
    crosspol : bool
        Whether to solve for cross-polarization.
    norm_method : str or None
        Selected normalization method.

    Examples
    --------
    >>> solver = GainSubstitution(refant=0, phase_only=True, niter=50)
    >>> gains, wgt, resid = solver.solve(vis, flags, wgt, model, ...)
    """

    _SOLVER_NAME_ = "gain_substitution"

    _NORMALISER = {
        "median": np.median,
        "mean": np.mean,
    }

    def __init__(
        self,
        refant=0,
        phase_only=False,
        crosspol=False,
        normalise_gains=None,
        **kwargs,
    ):
        super(GainSubstitution, self).__init__(**kwargs)
        self.refant = refant
        self.phase_only = phase_only
        self.crosspol = crosspol
        self.norm_method = normalise_gains

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
        Run the gain substitution solver algorithm.

        This method prepares the visibility data (creating point source
        equivalents) and invokes the iterative substitution solver.

        Parameters
        ----------
        vis_vis : np.ndarray
            Complex observed visibilities. Shape: (ntime, nbl, nchan, npol).
        vis_flags : np.ndarray
            Boolean flags for observed visibilities (True is flagged).
        vis_weight : np.ndarray
            Weights for observed visibilities.
        model_vis : np.ndarray
            Complex model visibilities. Must be provided if `model_flags` is
            provided.
        model_flags : np.ndarray
            Boolean flags for model visibilities.
        gain_gain : np.ndarray
            Initial guess for complex gains. Shape:
            (ntime_sol, nchan_sol, nant, nrec, nrec).
        gain_weight : np.ndarray
            Weights for the gain solutions.
        gain_residual : np.ndarray
            Buffer to store residuals of the fit.
        ant1 : np.ndarray
            Indices of antenna 1 for each baseline.
        ant2 : np.ndarray
            Indices of antenna 2 for each baseline.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing (gain_gain, gain_weight, gain_residual) with
            the updated solutions.

        Raises
        ------
        ValueError
            If `refant` is out of bounds.
            If `model_vis` contains only zeros or is mismatched with flags.
        """
        if self.refant < 0 or self.refant >= gain_gain.shape[1]:
            raise ValueError(
                f"gain_substitution: Invalid refant: {self.refant}"
            )

        if model_vis is not None:
            if not np.max(np.abs(model_vis)) > 0.0:
                raise ValueError("gain_substitution: Model visibility is zero")

        if (model_vis is not None) ^ (model_flags is not None):
            raise ValueError(
                "gain_substitution: model_vis and model_flags "
                + "must both be provided or both be None"
            )

        (pointvis_vis, pointvis_weight) = create_point_vis(
            vis_vis, vis_flags, vis_weight, model_vis, model_flags
        )

        return gain_substitution(
            gain_gain,
            gain_weight,
            gain_residual,
            pointvis_vis,
            vis_flags,
            pointvis_weight,
            ant1,
            ant2,
            crosspol=self.crosspol,
            niter=self.niter,
            phase_only=self.phase_only,
            tol=self.tol,
            refant=self.refant,
        )

    def normalise_gains(self, gain: xr.DataArray) -> xr.DataArray:
        """
        Normalize gain amplitudes using the configured method.

        Parameters
        ----------
        gain : xarray.DataArray
            The gain array to normalize.

        Returns
        -------
        xarray.DataArray
            The normalized gain array.

        Raises
        ------
        ValueError
            If the `normalise_gains` method specified in `__init__` is
            not valid (i.e., not 'mean' or 'median').
        """
        if self.norm_method is None:
            return gain

        logger.info(f"Normalizing gains using {self.norm_method}")

        if self.norm_method not in self._NORMALISER:
            raise ValueError(
                f"Undefined normalisation function {self.norm_method}"
            )

        norm_func = self._NORMALISER[self.norm_method]

        gabs = norm_func(np.abs(gain))

        return gain / gabs
