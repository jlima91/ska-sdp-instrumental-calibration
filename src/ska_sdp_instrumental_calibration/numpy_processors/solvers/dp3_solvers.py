import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np
from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_func_python.visibility.operations import expand_polarizations

from ...data_managers.h5parm import read_h5parm_gains
from .dp3_threading import default_dp3_n_threads, get_dp3_lock, require_dp3
from .solver import Solver

logger = logging.getLogger(__name__)


__all__ = ["dp3_gaincal_solve", "Dp3GaincalSolver", "DP3ObservationInfo"]


_DP3_MODEL_DATA_NAME = "modeldata"
_DP3_SUPPORTED_CALTYPES = (
    "scalarphase",
    "diagonal",
    "diagonalamplitude",
    "diagonalphase",
    "fulljones",
)


class Dp3GaincalSolver(Solver):
    """
    Solver for antenna gains using the DP3 gaincal step.

    This class wraps :py:func:`dp3_gaincal_solve`, which streams the
    visibilities through an in-memory DP3 ``gaincal`` step. The model
    visibilities are always provided to DP3 (``gaincal.reusemodel``), so
    no sky model prediction happens inside DP3.

    The DP3 gaincal calibration type is derived from ``crosspol`` and
    ``phase_only``, see :py:func:`_get_dp3_caltype`.

    Requires the optional ``dp3`` package.

    Parameters
    ----------
    crosspol
        Solve for the cross-polarisation terms as well, i.e. the full
        Jones matrix. Default is False.
    phase_only
        Solve only for the phases of the diagonal terms. Can not be
        combined with ``crosspol``. Default is False.
    **kwargs
        Additional keyword arguments passed to the base `Solver` class
        (e.g., `niter`, `tol`). Unknown arguments are ignored.

    Attributes
    ----------
    crosspol : bool
        Whether the cross-polarisation terms are solved.
    phase_only : bool
        Whether only the phases are solved.
    caltype : str
        DP3 gaincal calibration type.

    Raises
    ------
    ImportError
        If the optional ``dp3`` package is not available.

    Examples
    --------
    >>> solver = Solver.get_solver("dp3_gaincal", crosspol=True)
    >>> gains, wgt, resid = solver.solve(vis, flags, wgt, model, ...)
    """

    _SOLVER_NAME_ = "dp3_gaincal"

    def __init__(
        self, crosspol: bool = False, phase_only: bool = False, **kwargs
    ):
        require_dp3("dp3_gaincal solver")
        super(Dp3GaincalSolver, self).__init__(**kwargs)
        self.crosspol = crosspol
        self.phase_only = phase_only
        self.caltype = _get_dp3_caltype(crosspol, phase_only)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the DP3 gaincal solver.

        Parameters
        ----------
        vis_vis
            Complex observed visibilities. Shape: (ntime, nbl, nchan, npol).
        vis_flags
            Boolean flags for observed visibilities (True is flagged).
        vis_weight
            Weights for observed visibilities.
        model_vis
            Complex model visibilities. Shape matches `vis_vis`.
        model_flags
            Boolean flags for model visibilities.
        gain_gain
            Initial guess for complex gains. Shape:
            (ntime_sol, nant, nchan_sol, nrec, nrec).
        gain_weight
            Weights for the gain solutions.
        gain_residual
            Buffer to store residuals of the fit. Returned unchanged, as
            DP3 does not report residuals.
        ant1
            Indices of antenna 1 for each baseline.
        ant2
            Indices of antenna 2 for each baseline.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing (gain_gain, gain_weight, gain_residual) with
            the updated solutions.

        Raises
        ------
        ValueError
            If `model_vis` is not provided.
        """
        if model_vis is None:
            raise ValueError("dp3_gaincal: model_vis must be provided")

        return dp3_gaincal_solve(
            vis_vis,
            vis_flags,
            vis_weight,
            model_vis,
            model_flags,
            gain_gain,
            gain_weight,
            gain_residual,
            ant1,
            ant2,
            caltype=self.caltype,
            niter=self.niter,
            tol=self.tol,
        )


@dataclass
class DP3ObservationInfo:
    """
    Observation metadata required by DP3 to predict model visibilities.

    Parameters
    ----------
    antenna_names
        Names of the antennas. Shape: (nant,).
    antenna_positions
        ITRF XYZ antenna positions in meters. Shape: (nant, 3).
    antenna_diameters
        Antenna diameters in meters. Shape: (nant,).
    frequency
        Channel centre frequencies in Hz. Shape: (nfreq,).
    channel_bandwidth
        Channel widths in Hz. Shape: (nfreq,).
    time
        Centroid time of each timeslot in MJD seconds. Shape: (ntime,).
    integration_time
        Time interval between two timeslots in seconds.
    uvw
        Baseline UVW coordinates in meters. Shape: (ntime, nbl, 3).
    phasecentre
        Phase centre (ra, dec) in radians.
    """

    antenna_names: list[str]
    antenna_positions: np.ndarray
    antenna_diameters: np.ndarray
    frequency: np.ndarray
    channel_bandwidth: np.ndarray
    time: np.ndarray
    integration_time: float
    uvw: np.ndarray
    phasecentre: tuple[float, float]

    @classmethod
    def from_visibility(cls, vis: Visibility) -> "DP3ObservationInfo":
        """
        Create observation metadata from a Visibility.

        Parameters
        ----------
        vis
            Visibility to extract the metadata from.

        Returns
        -------
        DP3ObservationInfo
            Observation metadata of the visibility.
        """
        return cls(
            antenna_names=list(vis.configuration.names.data),
            antenna_positions=vis.configuration.xyz.data,
            antenna_diameters=vis.configuration.diameter.data,
            frequency=vis.frequency.data,
            channel_bandwidth=vis.channel_bandwidth.data,
            time=vis.time.data,
            integration_time=float(vis.integration_time.data[0]),
            uvw=vis.uvw.data,
            phasecentre=(vis.phasecentre.ra.rad, vis.phasecentre.dec.rad),
        )


def dp3_gaincal_solve(
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
    *,
    caltype: str = "diagonal",
    niter: int = 50,
    tol: float = 1e-6,
    skymodel_path: str | None = None,
    observation: DP3ObservationInfo | None = None,
    n_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for antenna gains using the DP3 gaincal step in memory.

    The visibilities are streamed one timeslot at a time through a DP3
    ``gaincal`` step created via the python bindings. If ``model_vis`` is
    given, it is supplied as extra buffer data (``gaincal.reusemodel``).
    Otherwise DP3 predicts the model internally from ``skymodel_path``
    (``gaincal.sourcedb``), which requires the ``observation`` metadata.
    The solution intervals in time and frequency are derived from the
    shape of ``gain_gain``.

    DP3 can only write solutions to an h5parm file, so the solutions are
    written to a temporary file and read back.

    Parameters
    ----------
    vis_vis
        Observed visibility data. Shape: (time, baseline, freq, pol).
    vis_flags
        Flags for observed visibilities. Shape matches `vis_vis`.
    vis_weight
        Weights for observed visibilities. Shape matches `vis_vis`.
    model_vis
        Model visibility data. Shape matches `vis_vis`.
        If None, DP3 predicts the model from ``skymodel_path``.
    model_flags
        Flags for model visibilities. Shape matches `vis_vis`.
        Flagged model samples are also flagged in the observed data.
    gain_gain
        Initial gain estimates. Shape: (time, ant, freq, rec1, rec2).
        Only the shape and the receptor terms not solved by DP3 for the
        given ``caltype`` are used; DP3 starts from its own initial guess.
    gain_weight
        Initial gain weights. Shape matches `gain_gain`.
        Receptor terms not solved by DP3 for the given ``caltype`` keep
        these weights.
    gain_residual
        Storage for gain residuals. Shape matches `gain_gain`.
        DP3 does not report residuals, so this is returned unchanged.
    ant1
        Indices of antenna 1 for each baseline. Shape: (nbl,).
    ant2
        Indices of antenna 2 for each baseline. Shape: (nbl,).
    caltype
        DP3 gaincal calibration type, e.g. ``diagonal``, ``scalarphase``
        or ``fulljones``.
    niter
        Maximum number of solver iterations (``gaincal.maxiter``).
    tol
        Solver convergence tolerance (``gaincal.tolerance``).
    skymodel_path
        Path to the DP3 sky model used to predict the model visibilities.
        Required if ``model_vis`` is None, ignored otherwise.
    observation
        Observation metadata. Required if ``model_vis`` is None. If not
        given while ``model_vis`` is provided, placeholder metadata is
        used.
    n_threads
        Number of threads for DP3's internal thread pool. If None, the
        number of threads of the current dask worker is used, or all
        available CPUs when not running on a worker.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:

        - Updated gain array.
        - Gain weights.
        - Gain residuals.

    Raises
    ------
    ImportError
        If the optional ``dp3`` package is not available.
    ValueError
        If neither a model nor a sky model with observation metadata is
        provided, ``caltype`` is not supported or the gain shape is
        incompatible with the visibility shape.
    """
    require_dp3("dp3_gaincal_solve")

    import dp3  # pylint: disable=import-error,import-outside-toplevel
    from dp3.parameterset import (  # pylint: disable=import-error,import-outside-toplevel # noqa: E501
        ParameterSet,
    )

    if model_vis is None and (skymodel_path is None or observation is None):
        raise ValueError(
            "dp3_gaincal: skymodel_path and observation are required "
            "when model_vis is not provided"
        )
    if caltype not in _DP3_SUPPORTED_CALTYPES:
        raise ValueError(
            f"dp3_gaincal: unsupported caltype {caltype!r}. "
            f"Supported: {', '.join(_DP3_SUPPORTED_CALTYPES)}"
        )

    n_time, n_baseline, n_freq, _ = vis_vis.shape
    n_soln_time, n_ant, n_soln_freq, _, _ = gain_gain.shape
    solint = _n_per_interval(n_time, n_soln_time, "time")
    nchan = _n_per_interval(n_freq, n_soln_freq, "frequency")

    flags = vis_flags if model_flags is None else vis_flags | model_flags

    dpinfo = _create_dp_info(ant1, ant2, n_ant, n_time, n_freq, observation)

    with tempfile.TemporaryDirectory() as tmpdir:
        h5parm_path = os.path.join(tmpdir, "gaincal.h5")

        parset = ParameterSet()
        parset.add("gaincal.parmdb", h5parm_path)
        if model_vis is None:
            parset.add("gaincal.sourcedb", skymodel_path)
        else:
            parset.add("gaincal.reusemodel", _DP3_MODEL_DATA_NAME)
        parset.add("gaincal.caltype", caltype)
        parset.add("gaincal.solint", str(solint))
        parset.add("gaincal.nchan", str(nchan))
        parset.add("gaincal.maxiter", str(niter))
        parset.add("gaincal.tolerance", str(tol))

        # Serialise all DP3 calls within this process, see get_dp3_lock
        with get_dp3_lock():
            # DP3's thread pool is process-wide, so set it on every call
            dp3.set_n_threads(
                default_dp3_n_threads() if n_threads is None else n_threads
            )
            step = dp3.make_step(
                "gaincal", parset, "gaincal.", dp3.MsType.regular
            )
            step.set_next_step(
                dp3.make_step("null", ParameterSet(), "", dp3.MsType.regular)
            )
            step.set_info(dpinfo)

            zero_uvw = np.zeros((n_baseline, 3))
            for time_idx in range(n_time):
                dpbuffer = dp3.DPBuffer()
                if observation is None:
                    dpbuffer.set_time(time_idx + 0.5)
                    dpbuffer.set_uvw(zero_uvw)
                else:
                    dpbuffer.set_time(observation.time[time_idx])
                    # DP3 uses the opposite uvw sign convention
                    dpbuffer.set_uvw(-observation.uvw[time_idx])
                dpbuffer.set_data(
                    expand_polarizations(vis_vis[time_idx], np.complex64)
                )
                dpbuffer.set_weights(
                    expand_polarizations(vis_weight[time_idx], np.float32)
                )
                dpbuffer.set_flags(expand_polarizations(flags[time_idx], bool))
                if model_vis is not None:
                    dpbuffer.add_data(_DP3_MODEL_DATA_NAME)
                    dpbuffer.set_extra_data(
                        _DP3_MODEL_DATA_NAME,
                        expand_polarizations(
                            model_vis[time_idx], np.complex64
                        ),
                    )
                step.process(dpbuffer)
            step.finish()

        dp3_gain, weight, solved, _ = read_h5parm_gains(h5parm_path)

    _gain_gain = np.where(solved, dp3_gain, gain_gain).astype(gain_gain.dtype)
    _gain_weight = np.where(solved, weight, gain_weight).astype(
        gain_weight.dtype
    )

    return _gain_gain, _gain_weight, gain_residual.copy()


def _n_per_interval(n_samples: int, n_intervals: int, axis: str) -> int:
    """
    Number of samples per solution interval along one axis.

    Parameters
    ----------
    n_samples
        Number of visibility samples along the axis.
    n_intervals
        Number of solution intervals along the axis.
    axis
        Name of the axis, used in the error message.

    Returns
    -------
    int
        Number of samples per solution interval.

    Raises
    ------
    ValueError
        If the samples cannot be split into exactly ``n_intervals`` chunks.
    """
    n_per = int(np.ceil(n_samples / n_intervals))
    if int(np.ceil(n_samples / n_per)) != n_intervals:
        raise ValueError(
            f"dp3_gaincal: cannot split {n_samples} {axis} samples into "
            f"{n_intervals} solution intervals"
        )
    return n_per


def _create_dp_info(
    ant1: np.ndarray,
    ant2: np.ndarray,
    n_ant: int,
    n_time: int,
    n_freq: int,
    observation: DP3ObservationInfo | None,
):
    """
    Create the DP3 DPInfo describing the visibilities.

    Parameters
    ----------
    ant1
        Indices of antenna 1 for each baseline. Shape: (nbl,).
    ant2
        Indices of antenna 2 for each baseline. Shape: (nbl,).
    n_ant
        Number of antennas.
    n_time
        Number of timeslots.
    n_freq
        Number of frequency channels.
    observation
        Observation metadata. If None, placeholder metadata is used,
        which is only valid when DP3 does not predict the model.

    Returns
    -------
    dp3.DPInfo
        DPInfo with 4 correlations.
    """
    import dp3  # pylint: disable=import-error,import-outside-toplevel

    # gaincal only works with 4 correlations
    dpinfo = dp3.DPInfo(4)

    if observation is None:
        dpinfo.set_channels(
            np.arange(n_freq, dtype=float) + 1.0, np.ones(n_freq)
        )
        dpinfo.set_antennas(
            [f"ANT{idx}" for idx in range(n_ant)],
            np.ones(n_ant),
            np.zeros((n_ant, 3)),
            ant1.astype(int),
            ant2.astype(int),
        )
        dpinfo.set_times(0.5, n_time - 0.5, 1.0)
        dpinfo.phase_center = [0.0, 0.0]
    else:
        dpinfo.set_channels(
            observation.frequency, observation.channel_bandwidth
        )
        dpinfo.set_antennas(
            observation.antenna_names,
            observation.antenna_diameters,
            observation.antenna_positions,
            ant1.astype(int),
            ant2.astype(int),
        )
        dpinfo.set_times(
            observation.time[0],
            observation.time[-1],
            observation.integration_time,
        )
        dpinfo.phase_center = list(observation.phasecentre)

    return dpinfo


def _get_dp3_caltype(crosspol: bool, phase_only: bool) -> str:
    """
    Get the DP3 gaincal calibration type for the INST solver options.

    =========  ==========  =================
    crosspol   phase_only  caltype
    =========  ==========  =================
    False      False       ``diagonal``
    False      True        ``diagonalphase``
    True       False       ``fulljones``
    True       True        not supported
    =========  ==========  =================

    The scalar and amplitude-only caltypes of DP3 are not used, as the
    INST solver options have no equivalent for them. The ``tec`` and
    ``tec+phase`` caltypes are not supported, as INST handles the
    ionosphere with its own solvers.

    Parameters
    ----------
    crosspol
        Solve for the cross-polarisation terms as well.
    phase_only
        Solve only for the phases.

    Returns
    -------
    str
        DP3 gaincal calibration type.

    Raises
    ------
    ValueError
        If both ``crosspol`` and ``phase_only`` are set, as DP3 gaincal
        has no phase-only full-Jones calibration type.
    """
    if crosspol and phase_only:
        raise ValueError(
            "dp3_gaincal: phase_only can not be combined with crosspol, "
            "as DP3 gaincal has no phase-only fulljones caltype"
        )
    if crosspol:
        return "fulljones"
    return "diagonalphase" if phase_only else "diagonal"
