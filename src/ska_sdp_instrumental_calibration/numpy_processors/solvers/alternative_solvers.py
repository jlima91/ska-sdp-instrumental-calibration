import logging
from typing import Tuple

import numpy as np
from ska_sdp_func_python.calibration.alternative_solvers import (
    _jones_sub_solve,
    _normal_equation_solve,
    _normal_equation_solve_with_presumming,
)

from .solver import Solver

log = logging.getLogger()


class AlternativeSolver(Solver):
    """
    Base class for alternative antenna gain solvers using different algorithms.

    This class extends `Solver` to provide a flexible interface for
    alternative calibration algorithms, such as Jones substitution or
    normal equations. The specific algorithm is set via the `_solver_fn`
    property, which should return a callable implementing the desired method.
    """

    @property
    def _solver_fn(self):
        """
        callable: Function implementing the alternative solver algorithm.

        This property must be implemented by subclasses. The returned function
        should accept the observed visibilities, model visibilities, weights,
        antenna indices, and current gain estimates to perform the optimization

        Raises
        ------
        NotImplementedError
            If the subclass does not define this property.
        """
        raise NotImplementedError("Solver function not defined")

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
        Run the alternative gain solver algorithm.

        Fits observed visibilities to model visibilities using the selected
        algorithm (defined in `_solver_fn`). It handles reshaping of
        visibilities into Jones matrices, flag application, and model
        corruption by existing gains before invoking the specific solver.

        Parameters
        ----------
        vis_vis : np.ndarray
            Complex observed visibilities.
            Shape: (ntime, nbl, nchan, npol).
        vis_flags : np.ndarray
            Boolean flags for observed visibilities (True indicates flagged).
            Shape: (ntime, nbl, nchan, npol).
        vis_weight : np.ndarray
            Weights for observed visibilities.
            Shape: (ntime, nbl, nchan, npol).
        model_vis : np.ndarray
            Complex model visibilities.
            Shape: (ntime, nbl, nchan, npol).
        model_flags : np.ndarray
            Boolean flags for model visibilities.
            Shape: (ntime, nbl, nchan, npol).
        gain_gain : np.ndarray
            Complex gain solutions (Jones matrices). Acts as both the initial
            guess and the output buffer.
            Shape: (ntime_sol, nchan_sol, nant, nrec, nrec).
        gain_weight : np.ndarray
            Weights associated with the gain solutions.
            Shape: matches `gain_gain`.
        gain_residual : np.ndarray
            Residuals of the gain solutions.
            Shape: matches `gain_gain`.
        ant1 : np.ndarray
            Index of the first antenna for each baseline.
            Shape: (nbl,).
        ant2 : np.ndarray
            Index of the second antenna for each baseline.
            Shape: (nbl,).

        Returns
        -------
        tuple of np.ndarray
            A tuple containing:

            - `gain_gain`: Updated complex gain solutions.
            - `gain_weight`: Weights of the solutions.
            - `gain_residual`: Residuals of the fit.
        """
        _gain_gain = gain_gain.copy()
        gain = _gain_gain[0]  # select first time

        _, nchan_gt, nrec1, nrec2 = gain.shape
        ntime, nbl, nchan_vis, npol_vis = vis_vis.shape
        assert nrec1 == 2
        assert nrec1 == nrec2
        assert nrec1 * nrec2 == npol_vis
        assert nchan_gt in (1, nchan_vis)

        # incorporate flags into weights
        wgt = vis_weight * (1 - vis_flags)
        # flag the whole Jones matrix if any element is flagged
        wgt *= np.all(wgt > 0, axis=-1, keepdims=True)
        # reduce the dimension to a single weight per matrix
        #   - could weight pols separately, but may be better not to
        wgt = wgt[..., 0]

        vmdl = model_vis.reshape(ntime, nbl, nchan_vis, 2, 2)
        vobs = vis_vis.reshape(ntime, nbl, nchan_vis, 2, 2)

        # Update model if a starting solution is given.
        I2 = np.eye(2)
        if np.any(gain[..., :, :] != I2):
            vmdl = np.einsum(
                "bfpi,tbfij,bfqj->tbfpq",
                gain[ant1],
                vmdl,
                gain[ant2].conj(),
            )

        log.debug(
            "solve_with_alternative_algorithm: "
            + "solving for %d chan in %d sub-band[s] using solver %s",
            nchan_vis,
            nchan_gt,
            self._solver_fn,
        )

        for ch in range(nchan_gt):
            # select channels to average over. Just the current one
            # if solving each channel separately, or all of them
            # if this is a joint solution.
            chan_vis = [ch] if nchan_gt == nchan_vis else range(nchan_vis)

            log.debug(
                "solve_with_alternative_algorithm: "
                + "sub-band %d, processing %d channels:",
                ch,
                len(chan_vis),
            )

            self._solver_fn(  # pylint: disable=not-callable
                vobs[:, :, chan_vis],
                vmdl[:, :, chan_vis],
                wgt[:, :, chan_vis],
                ant1,
                ant2,
                gain,
                ch,
                self.niter,
                self.tol,
            )

        _gain_gain[0, ...] = gain

        return (_gain_gain, gain_weight, gain_residual)


class JonesSubtitution(AlternativeSolver):
    """
    Solver for antenna gains using the Jones substitution algorithm.

    This class configures the `AlternativeSolver` to use the Jones matrix
    substitution method for calibration. It sets the internal solver function
    to an implementation that iteratively solves for antenna-based Jones
    matrices.

    Attributes
    ----------
    _SOLVER_NAME_ : str
        The unique identifier for this solver strategy ("jones_substitution").

    Notes
    -----
    This class is typically used for full-polarization calibration where
    antenna-based Jones matrices are solved iteratively.
    """

    _SOLVER_NAME_ = "jones_substitution"

    @property
    def _solver_fn(self):
        """
        callable: The Jones substitution solver function.

        Returns `_jones_sub_solve`, which implements the specific algebraic
        substitution logic.
        """
        return _jones_sub_solve


class NormalEquation(AlternativeSolver):
    """
    Solver for antenna gains using Normal Equations.

    This class configures the `AlternativeSolver` to use a Normal Equations
    approach (Linear Least Squares) to determine antenna gains. It provides
    a robust method for solving complex gain matrices by minimizing the
    difference between model and observed visibilities.

    Attributes
    ----------
    _SOLVER_NAME_ : str
        The unique identifier for this solver strategy ("normal_equations").
    """

    _SOLVER_NAME_ = "normal_equations"

    @property
    def _solver_fn(self):
        """
        callable: The Normal Equation solver function.

        Returns `_normal_equation_solve`, which constructs and solves the
        system of linear equations.
        """
        return _normal_equation_solve


class NormalEquationsPreSum(AlternativeSolver):
    """
    Solver using Normal Equations with data pre-summing.

    This class extends the Normal Equations approach by applying a pre-summing
    optimization. Visibility data is averaged (pre-summed) before the
    iterative solving steps to reduce computational overhead on large
    datasets while maintaining solution accuracy.

    Attributes
    ----------
    _SOLVER_NAME_ : str
        The unique identifier for this solver strategy
        ("normal_equations_presum").
    """

    _SOLVER_NAME_ = "normal_equations_presum"

    @property
    def _solver_fn(self):
        """
        callable: The Pre-summed Normal Equation solver function.

        Returns `_normal_equation_solve_with_presumming`, which handles the
        data reduction and subsequent solving steps.
        """
        return _normal_equation_solve_with_presumming
