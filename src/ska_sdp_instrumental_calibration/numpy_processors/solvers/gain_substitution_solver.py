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
    antenna-based complex gains in radio interferometric calibration.

    Parameters
    ----------
    refant : int, optional
        Index of the reference antenna. (default is 0)
    phase_only : bool, optional
        If True, solve for phase-only gains. If False, solve for complex gains.
        (default is False)
    crosspol : bool, optional
        If True, solve for cross-polarization terms as well. (default is False)
        i.e. XY, YX or RL, LR.

    Examples
    --------
    >>> solver = GainSubstitution(refant=0, phase_only=True, crosspol=False,
    ...     normalise_gains="mean", niter=100, tol=1e-6)
    >>> updated_gains = solver.solve(
    ...     vis_vis, vis_flags, vis_weight, model_vis, model_flags,
    ...     gain_gain, gain_weight, gain_residual, ant1, ant2
    ... )
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
        Run the gain solver algorithm.
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
        Function to normalize gains

        Parameters
        ----------
        gaintable: xarray.DataArray

        Returns
        -------
        xarray.DataArray
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
