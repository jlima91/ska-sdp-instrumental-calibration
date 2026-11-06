"""
In-memory wrapper for DP3's direction-dependent calibration (DDECal) step.

Unlike dp3_gaincal_solve, this wrapper returns the path to a DP3 H5Parm
solution file. DDECal can solve separate gains for multiple sky directions,
so its result is not generally representable by the single gain array used
by the gaincal wrapper.

This implementation asks DP3/DDECal to predict model visibilities from a
DP3 sky-model file (sourcedb). It does not accept a precomputed model_vis
array and does not initialise DDECal from the gain_gain array used by
dp3_gaincal_solve.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from ska_sdp_func_python.visibility.operations import expand_polarizations

from ...data_managers.h5parm import read_h5parm_gains
from .dp3_solvers import (
    DP3ObservationInfo,
    _create_dp_info,
    _get_dp3_caltype,
    _n_per_interval,
)
from .dp3_threading import (
    default_dp3_n_threads,
    get_dp3_lock,
    require_dp3,
    set_dp3_n_threads,
)
from .solver import Solver

__all__ = ["dp3_ddecal_solve", "Dp3DdecalSolver"]


class Dp3DdecalSolver(Solver):
    """
    Solver for antenna gains using the DP3 DDECal step.

    This class wraps :py:func:`dp3_ddecal_solve`. The DP3 6.x Python
    bindings cannot hand in-memory model visibilities to DDECal, so DDECal
    predicts the model itself from ``skymodel_path``; the ``model_vis``
    passed to :py:meth:`solve` is ignored (``model_flags`` are still
    applied). Only single-direction sky models are supported, as the
    solutions are returned as a single gain array.

    Because DDECal predicts the model, :py:meth:`solve` needs the
    observation metadata (time, UVW, frequency, antennas) of the chunk.
    ``requires_observation`` tells :py:func:`run_solver` to provide it.

    The DDECal mode is derived from ``crosspol`` and ``phase_only``, see
    :py:func:`_get_dp3_caltype`.

    Requires the optional ``dp3`` package.

    Parameters
    ----------
    skymodel_path
        DP3-readable sky model (path ending in ``.skymodel`` or ``.txt``,
        or a sourcedb directory).
    crosspol
        Solve for the cross-polarisation terms as well, i.e. the full
        Jones matrix. Default is False.
    phase_only
        Solve only for the phases of the diagonal terms. Can not be
        combined with ``crosspol``. Default is False.
    n_threads
        DP3 thread count; None uses the project's default thread policy.
    **kwargs
        Additional keyword arguments passed to the base `Solver` class
        (e.g., `niter`, `tol`). Unknown arguments are ignored.

    Examples
    --------
    >>> solver = Solver.get_solver(
    ...     "dp3_ddecal", skymodel_path="sky.skymodel", crosspol=True
    ... )
    >>> gaintable = run_solver(vis, modelvis, gaintable, solver)
    """

    _SOLVER_NAME_ = "dp3_ddecal"
    requires_observation = True

    def __init__(
        self,
        skymodel_path: str | None = None,
        crosspol: bool = False,
        phase_only: bool = False,
        n_threads: int | None = None,
        **kwargs,
    ):
        require_dp3("dp3_ddecal solver")
        super().__init__(**kwargs)
        if not skymodel_path:
            raise ValueError("dp3_ddecal: skymodel_path must be provided")
        self.skymodel_path = skymodel_path
        self.crosspol = crosspol
        self.phase_only = phase_only
        self.n_threads = n_threads
        # DDECal accepts the same mode names as the gaincal caltypes used here
        self.mode = _get_dp3_caltype(crosspol, phase_only)

    def solve(
        self,
        vis_vis: np.ndarray,
        vis_flags: np.ndarray,
        vis_weight: np.ndarray,
        model_vis: np.ndarray | None,
        model_flags: np.ndarray | None,
        gain_gain: np.ndarray,
        gain_weight: np.ndarray,
        gain_residual: np.ndarray,
        ant1: np.ndarray,
        ant2: np.ndarray,
        observation: DP3ObservationInfo | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the DP3 DDECal solver.

        Parameters
        ----------
        vis_vis, vis_flags, vis_weight
            Observed visibilities, flags and weights.
            Shape: (ntime, nbl, nchan, npol).
        model_vis
            Ignored, DDECal predicts the model from ``skymodel_path``.
        model_flags
            Flags of the model visibilities, combined with ``vis_flags``.
        gain_gain
            Initial gains. Shape: (ntime_sol, nant, nchan_sol, nrec, nrec).
            Solution intervals are derived from its shape, and its values
            are kept where DDECal did not solve.
        gain_weight
            Gain weights. Shape matches ``gain_gain``.
        gain_residual
            Returned unchanged, as DP3 does not report residuals.
        ant1, ant2
            Antenna indices for each baseline.
        observation
            Metadata describing the same time, frequency, antenna and UVW
            axes as ``vis_vis``. Provided by :py:func:`run_solver`.

        Returns
        -------
        tuple of np.ndarray
            (gain_gain, gain_weight, gain_residual) with the updated
            solutions.
        """
        if observation is None:
            raise ValueError(
                "dp3_ddecal: observation metadata must be provided"
            )

        n_time, _, n_freq, _ = vis_vis.shape
        n_soln_time, _, n_soln_freq, _, _ = gain_gain.shape
        solint = _n_per_interval(n_time, n_soln_time, "time")
        nchan = _n_per_interval(n_freq, n_soln_freq, "frequency")

        flags = vis_flags if model_flags is None else vis_flags | model_flags

        with tempfile.TemporaryDirectory() as tmpdir:
            h5parm_path = dp3_ddecal_solve(
                vis_vis,
                flags,
                vis_weight,
                ant1,
                ant2,
                skymodel_path=self.skymodel_path,
                observation=observation,
                output_h5parm_path=os.path.join(tmpdir, "ddecal.h5"),
                mode=self.mode,
                solint=solint,
                nchan=nchan,
                niter=self.niter,
                tol=self.tol,
                n_threads=self.n_threads,
            )
            dp3_gain, weight, solved, _ = read_h5parm_gains(h5parm_path)

        # Keep the initial values for terms DDECal did not solve
        _gain_gain = np.where(solved, dp3_gain, gain_gain).astype(
            gain_gain.dtype
        )
        _gain_weight = np.where(solved, weight, gain_weight).astype(
            gain_weight.dtype
        )

        return _gain_gain, _gain_weight, gain_residual.copy()


def dp3_ddecal_solve(
    vis_vis: np.ndarray,
    vis_flags: np.ndarray,
    vis_weight: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    *,
    skymodel_path: str,
    observation: DP3ObservationInfo,
    output_h5parm_path: str,
    mode: str = "fulljones",
    solint: int = 0,
    nchan: int = 1,
    niter: int = 50,
    tol: float = 1e-5,
    n_threads: int | None = None,
) -> str:
    """
    Run DP3's DDECal step on in-memory visibility arrays.

    Parameters
    ----------
    vis_vis
        Observed complex visibilities, shape (ntime, nbaseline, nfreq, npol).
        The current DP3 wrapper expects four correlations.
    vis_flags
        Boolean visibility flags; True means flagged. Same shape as vis_vis.
    vis_weight
        Visibility weights. Same shape as vis_vis.
    ant1, ant2
        Antenna indices for each baseline, each with shape (nbaseline,).
    skymodel_path
        DP3-readable sky model (sourcedb). DDECal predicts the direction
        models from this sky model.
    observation
        Metadata produced by DP3ObservationInfo.from_visibility(...).
        Must describe the same time, frequency, antenna and UVW axes as
        the supplied arrays.
    output_h5parm_path
        Destination for DDECal's H5Parm solution file. Parent directories
        are created if needed.
    mode
        DDECal solve mode, e.g. "diagonal" or "fulljones". The chosen mode
        must be supported by the installed DP3 version.
    solint
        Solution interval in timeslots. Zero means one interval covering
        the whole observation.
    nchan
        Number of channels per solution/channel block. Zero means one
        solution across the full frequency range.
    niter
        Maximum DDECal solver iterations.
    tol
        DDECal convergence tolerance.
    n_threads
        DP3 thread count; None uses the project's default thread policy.

    Returns
    -------
    str
        Absolute path to the H5Parm file written by DDECal.

    Notes
    -----
    This intentionally returns an H5Parm path rather than
    (gain, weight, residual). DDECal solutions may have a direction axis
    and DP3 writes them to H5Parm. The existing read_h5parm_gains helper
    should not be assumed to preserve all directions without checking its
    implementation and the generated file's soltabs.
    """
    require_dp3("dp3_ddecal_solve")

    import dp3  # pylint: disable=import-error,import-outside-toplevel

    # pylint: disable=import-error,import-outside-toplevel
    from dp3.parameterset import ParameterSet

    if not skymodel_path:
        raise ValueError("dp3_ddecal: skymodel_path must be provided")
    if observation is None:
        raise ValueError("dp3_ddecal: observation metadata must be provided")
    if not output_h5parm_path:
        raise ValueError("dp3_ddecal: output_h5parm_path must be provided")
    if mode not in {
        "scalar",
        "scalarphase",
        "scalaramplitude",
        "diagonal",
        "diagonalphase",
        "diagonalamplitude",
        "fulljones",
        "tec",
        "tec+delay",
        "tec+phase",
        "tec+phase+delay",
        "rotation",
        "rotation+diagonal",
        "faradayrotation",
        "leakage",
        "leakageamplitude",
    }:
        raise ValueError(f"dp3_ddecal: unsupported mode {mode!r}")

    vis_vis = np.asarray(vis_vis)
    vis_flags = np.asarray(vis_flags, dtype=bool)
    vis_weight = np.asarray(vis_weight)

    if vis_vis.ndim != 4:
        raise ValueError(
            "dp3_ddecal: vis_vis must have shape "
            "(ntime, nbaseline, nfreq, npol)"
        )
    if vis_vis.shape[-1] != 4:
        raise ValueError(
            "dp3_ddecal: expected four correlations in the last axis"
        )
    if vis_flags.shape != vis_vis.shape or vis_weight.shape != vis_vis.shape:
        raise ValueError(
            "dp3_ddecal: vis_flags and vis_weight must match vis_vis shape"
        )

    n_time, n_baseline, n_freq, _ = vis_vis.shape
    ant1 = np.asarray(ant1)
    ant2 = np.asarray(ant2)
    if ant1.shape != (n_baseline,) or ant2.shape != (n_baseline,):
        raise ValueError(
            "dp3_ddecal: ant1 and ant2 must each have shape (nbaseline,)"
        )
    if len(observation.time) != n_time:
        raise ValueError(
            "dp3_ddecal: observation time axis does not match vis_vis"
        )
    if len(observation.frequency) != n_freq:
        raise ValueError(
            "dp3_ddecal: observation frequency axis does not match vis_vis"
        )
    if observation.uvw.shape[:2] != (n_time, n_baseline):
        raise ValueError(
            "dp3_ddecal: observation UVW axes do not match vis_vis"
        )

    if solint < 0:
        raise ValueError("dp3_ddecal: solint must be >= 0")
    if nchan < 0:
        raise ValueError("dp3_ddecal: nchan must be >= 0")
    if niter < 1:
        raise ValueError("dp3_ddecal: niter must be >= 1")
    if tol <= 0:
        raise ValueError("dp3_ddecal: tol must be > 0")

    output_path = Path(output_h5parm_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dpinfo = _create_dp_info(
        ant1, ant2, len(observation.antenna_names), n_time, n_freq, observation
    )

    parset = ParameterSet()
    parset.add(
        "ddecal.sourcedb",
        os.fspath(Path(skymodel_path).expanduser().resolve()),
    )
    parset.add("ddecal.h5parm", os.fspath(output_path))
    parset.add("ddecal.mode", mode)
    parset.add("ddecal.solint", str(solint))
    parset.add("ddecal.nchan", str(nchan))
    parset.add("ddecal.maxiter", str(niter))
    parset.add("ddecal.tolerance", str(tol))

    # DDECal predicts per-direction models from sourcedb. DPBuffer provides
    # the observed data, flags, weights, time and UVW for each timeslot.
    with get_dp3_lock():
        set_dp3_n_threads(
            default_dp3_n_threads() if n_threads is None else n_threads
        )
        step = dp3.make_step("ddecal", parset, "ddecal.", dp3.MsType.regular)
        step.set_next_step(
            dp3.make_step("null", ParameterSet(), "", dp3.MsType.regular)
        )
        step.set_info(dpinfo)

        for time_idx in range(n_time):
            dpbuffer = dp3.DPBuffer()
            dpbuffer.set_time(float(observation.time[time_idx]))
            # DP3 uses the opposite UVW sign convention.
            dpbuffer.set_uvw(-observation.uvw[time_idx])
            dpbuffer.set_data(
                expand_polarizations(vis_vis[time_idx], np.complex64)
            )
            dpbuffer.set_weights(
                expand_polarizations(vis_weight[time_idx], np.float32)
            )
            dpbuffer.set_flags(expand_polarizations(vis_flags[time_idx], bool))
            step.process(dpbuffer)

        step.finish()

    return os.fspath(output_path)
