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
    normal equations. The specific algorithm is set via the `solver_fn`
    attribute, which should be a callable implementing the desired method.

    Attributes
    ----------
    solver_fn : callable
        Function implementing the alternative solver algorithm.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._solver_fn = None

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
        algorithm.
        """
        if self._solver_fn is None:
            raise ValueError(
                "AlternativeSolver: alternative solver function to "
                + "be used is not provided."
            )

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
        #  - could weight pols separately, but may be better not to
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
    substitution method for calibration. It sets the `solver_fn` attribute
    to the appropriate Jones substitution function.

    Notes
    -----
    This class is typically used for full-polarization calibration where
    antenna-based Jones matrices are solved iteratively.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver_fn = _jones_sub_solve


class NormalEquation(AlternativeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver_fn = _normal_equation_solve


class NormalEquationsPreSum(AlternativeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver_fn = _normal_equation_solve_with_presumming
