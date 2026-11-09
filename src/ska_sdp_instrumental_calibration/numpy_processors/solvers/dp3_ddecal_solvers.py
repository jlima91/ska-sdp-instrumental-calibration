"""
In-memory wrapper for DP3's direction-dependent calibration (DDECal) step.

:py:func:`dp3_ddecal_solve` follows the same contract as
:py:func:`dp3_gaincal_solve`: it takes the visibility, model and gain
arrays and returns ``(gain, weight, residual)``.

DDECal is always run for a single direction, so its solutions fit in the
single gain array. The model of that direction is either given as model
visibilities, or predicted by DDECal from a DP3 sky model: the patches
given by the user or, by default, the patch at the phase centre
(direction-independent solving).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ska_sdp_func_python.visibility.operations import expand_polarizations

from ...data_managers.h5parm import read_h5parm_gains
from .dp3_solvers import (
    _DP3_SUPPORTED_CALTYPES,
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

logger = logging.getLogger(__name__)

# Prefix of the DDECal step that registers the model direction, and the
# name of the model data; DDECal reuses the model "<prefix><name>"
_DP3_MODEL_STEP_PREFIX = "modelinfo."
_DP3_MODEL_DATA_NAME = "modeldata"


class Dp3DdecalSolver(Solver):
    """
    Solver for antenna gains using the DP3 DDECal step.

    This class wraps :py:func:`dp3_ddecal_solve`, solving for a single
    direction. By default the ``model_vis`` passed to :py:meth:`solve` is
    used as the model. If ``skymodel_path`` is given, DDECal predicts the
    model from that sky model instead and ``model_vis`` is ignored
    (``model_flags`` are still applied).

    :py:meth:`solve` takes the observation metadata (time, UVW,
    frequency, antennas) of the chunk, which DDECal needs to predict the
    model. ``requires_observation`` tells :py:func:`run_solver` to provide
    it.

    The DDECal mode is derived from ``crosspol`` and ``phase_only``, see
    :py:func:`_get_dp3_caltype`.

    Requires the optional ``dp3`` package.

    Parameters
    ----------
    skymodel_path
        Optional DP3-readable sky model (path ending in ``.skymodel`` or
        ``.txt``, or a sourcedb directory) to predict the model from.
        None uses the model visibilities.
    direction
        Sky model patches combined into the single solved direction, only
        used with ``skymodel_path``. None solves direction independently
        for the patch nearest to the phase centre, see
        :py:func:`dp3_ddecal_solve`.
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
    >>> solver = Solver.get_solver("dp3_ddecal", crosspol=True)
    >>> gaintable = run_solver(vis, modelvis, gaintable, solver)
    """

    _SOLVER_NAME_ = "dp3_ddecal"
    requires_observation = True

    def __init__(
        self,
        skymodel_path: str | None = None,
        direction: Sequence[str] | None = None,
        crosspol: bool = False,
        phase_only: bool = False,
        n_threads: int | None = None,
        **kwargs,
    ):
        require_dp3("dp3_ddecal solver")
        super().__init__(**kwargs)
        self.skymodel_path = skymodel_path
        self.direction = direction
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

        See :py:func:`dp3_ddecal_solve` for the parameters. ``observation``
        is provided by :py:func:`run_solver`. ``model_vis`` is not used if
        the solver has a ``skymodel_path``.

        Returns
        -------
        tuple of np.ndarray
            (gain_gain, gain_weight, gain_residual) with the updated
            solutions.
        """
        return dp3_ddecal_solve(
            vis_vis,
            vis_flags,
            vis_weight,
            # With a sky model, DDECal predicts the model itself
            None if self.skymodel_path else model_vis,
            model_flags,
            gain_gain,
            gain_weight,
            gain_residual,
            ant1,
            ant2,
            skymodel_path=self.skymodel_path,
            observation=observation,
            direction=self.direction,
            mode=self.mode,
            niter=self.niter,
            tol=self.tol,
            n_threads=self.n_threads,
        )


def dp3_ddecal_solve(
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
    skymodel_path: str | None = None,
    observation: DP3ObservationInfo | None = None,
    direction: Sequence[str] | None = None,
    mode: str = "diagonal",
    niter: int = 50,
    tol: float = 1e-6,
    n_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for antenna gains of a single direction using DP3 DDECal.

    Same contract as :py:func:`dp3_gaincal_solve`. The visibilities are
    streamed one timeslot at a time through a DP3 ``ddecal`` step created
    via the python bindings. If ``model_vis`` is given, it is supplied as
    extra buffer data (``ddecal.reusemodel``). Otherwise DDECal predicts
    the model from ``skymodel_path`` (``ddecal.sourcedb``), which requires
    the ``observation`` metadata. The solution intervals in time and
    frequency are derived from the shape of ``gain_gain``.

    DP3 can only write solutions to an h5parm file, so the solutions are
    written to a temporary file and read back.

    Parameters
    ----------
    vis_vis
        Observed visibility data. Shape: (time, baseline, freq, pol),
        with four polarisations.
    vis_flags
        Flags for observed visibilities. Shape matches `vis_vis`.
    vis_weight
        Weights for observed visibilities. Shape matches `vis_vis`.
    model_vis
        Model visibility data of the solved direction. Shape matches
        `vis_vis`. If None, DDECal predicts the model from
        ``skymodel_path``.
    model_flags
        Flags for model visibilities. Shape matches `vis_vis`.
        Flagged model samples are also flagged in the observed data.
    gain_gain
        Initial gain estimates. Shape: (time, ant, freq, rec1, rec2).
        Only the shape and the receptor terms not solved by DP3 for the
        given ``mode`` are used; DP3 starts from its own initial guess.
    gain_weight
        Initial gain weights. Shape matches `gain_gain`.
        Receptor terms not solved by DP3 for the given ``mode`` keep
        these weights.
    gain_residual
        Storage for gain residuals. Shape matches `gain_gain`.
        DP3 does not report residuals, so this is returned unchanged.
    ant1
        Indices of antenna 1 for each baseline. Shape: (nbl,).
    ant2
        Indices of antenna 2 for each baseline. Shape: (nbl,).
    skymodel_path
        DP3 sky model used to predict the model visibilities (path ending
        in ``.skymodel`` or ``.txt``, or a sourcedb directory).
        Required if ``model_vis`` is None, ignored otherwise.
    observation
        Metadata describing the same time, frequency, antenna and UVW
        axes as ``vis_vis``, e.g. from
        :py:meth:`DP3ObservationInfo.from_visibility`. Required if
        ``model_vis`` is None. If not given while ``model_vis`` is
        provided, placeholder metadata is used.
    direction
        Only used if ``model_vis`` is None. Names of the sky model patches
        that form the single solved direction
        (``ddecal.directions=[[...]]``). For a sky model without patches,
        each source is its own patch, named after the source. None solves
        direction independently: only the patch nearest to the phase
        centre of ``observation`` is used, which needs a text sky model
        with positions in angle units (e.g. ``deg``). A patch's direction
        is the flux-weighted mean of its sources, as in DP3.
    mode
        DDECal solve mode, e.g. ``diagonal``, ``scalarphase`` or
        ``fulljones``.
    niter
        Maximum number of solver iterations (``ddecal.maxiter``).
    tol
        Solver convergence tolerance (``ddecal.tolerance``).
    n_threads
        Number of threads for DP3's internal thread pool. If None, it is
        chosen by :py:func:`default_dp3_n_threads`.

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
        provided, the direction is invalid, ``mode`` is not supported or
        the array shapes are inconsistent.
    """
    require_dp3("dp3_ddecal_solve")

    import dp3  # pylint: disable=import-error,import-outside-toplevel
    from dp3.parameterset import (  # pylint: disable=import-error,import-outside-toplevel # noqa: E501
        ParameterSet,
    )

    if model_vis is None and (not skymodel_path or observation is None):
        raise ValueError(
            "dp3_ddecal: skymodel_path and observation are required "
            "when model_vis is not provided"
        )
    if mode not in _DP3_SUPPORTED_CALTYPES:
        raise ValueError(
            f"dp3_ddecal: unsupported mode {mode!r}. "
            f"Supported: {', '.join(_DP3_SUPPORTED_CALTYPES)}"
        )

    if vis_vis.ndim != 4 or vis_vis.shape[-1] != 4:
        raise ValueError(
            "dp3_ddecal: vis_vis must have shape "
            "(ntime, nbaseline, nfreq, 4)"
        )
    if vis_flags.shape != vis_vis.shape or vis_weight.shape != vis_vis.shape:
        raise ValueError(
            "dp3_ddecal: vis_flags and vis_weight must match vis_vis shape"
        )
    if model_vis is not None and model_vis.shape != vis_vis.shape:
        raise ValueError("dp3_ddecal: model_vis must match vis_vis shape")

    n_time, n_baseline, n_freq, _ = vis_vis.shape
    n_soln_time, n_ant, n_soln_freq, _, _ = gain_gain.shape
    solint = _n_per_interval(n_time, n_soln_time, "time")
    nchan = _n_per_interval(n_freq, n_soln_freq, "frequency")

    if observation is not None:
        if len(observation.time) != n_time:
            raise ValueError(
                "dp3_ddecal: observation time axis does not match vis_vis"
            )
        if len(observation.frequency) != n_freq:
            raise ValueError(
                "dp3_ddecal: observation frequency axis does not match "
                "vis_vis"
            )
        if observation.uvw.shape[:2] != (n_time, n_baseline):
            raise ValueError(
                "dp3_ddecal: observation UVW axes do not match vis_vis"
            )

    patches = None
    if model_vis is None:
        skymodel_path = os.fspath(Path(skymodel_path).expanduser().resolve())
        # Without a direction, solve direction independently: only the
        # patch at the phase centre is in the model, all others are left out
        patches = (
            [_find_phase_centre_patch(skymodel_path, observation.phasecentre)]
            if direction is None
            else list(direction)
        )
        if not patches or not all(isinstance(p, str) and p for p in patches):
            raise ValueError(
                "dp3_ddecal: direction must be a non-empty list of patch "
                "names"
            )

    flags = vis_flags if model_flags is None else vis_flags | model_flags

    dpinfo = _create_dp_info(ant1, ant2, n_ant, n_time, n_freq, observation)
    model_name = _DP3_MODEL_STEP_PREFIX + _DP3_MODEL_DATA_NAME

    with tempfile.TemporaryDirectory() as tmpdir:
        h5parm_path = os.path.join(tmpdir, "ddecal.h5")

        parset = ParameterSet()
        if model_vis is None:
            parset.add("ddecal.sourcedb", skymodel_path)
            # A single direction holding all given patches. Without this,
            # DDECal solves one direction per patch of the sky model.
            parset.add("ddecal.directions", f"[[{','.join(patches)}]]")
        else:
            parset.add("ddecal.reusemodel", f"[{model_name}]")
        parset.add("ddecal.h5parm", h5parm_path)
        parset.add("ddecal.mode", mode)
        parset.add("ddecal.solint", str(solint))
        parset.add("ddecal.nchan", str(nchan))
        parset.add("ddecal.maxiter", str(niter))
        parset.add("ddecal.tolerance", str(tol))

        # Serialise all DP3 calls within this process, see get_dp3_lock
        with get_dp3_lock():
            # DP3's thread pool is process-wide, so check it on every call
            set_dp3_n_threads(
                default_dp3_n_threads() if n_threads is None else n_threads
            )
            if model_vis is not None:
                dpinfo = _register_model_direction(
                    dpinfo,
                    (
                        (0.0, 0.0)
                        if observation is None
                        else observation.phasecentre
                    ),
                    tmpdir,
                )
            step = dp3.make_step(
                "ddecal", parset, "ddecal.", dp3.MsType.regular
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
                    dpbuffer.add_data(model_name)
                    dpbuffer.set_extra_data(
                        model_name,
                        expand_polarizations(
                            model_vis[time_idx], np.complex64
                        ),
                    )
                step.process(dpbuffer)
            step.finish()
            # DDECal keeps its h5parm open until the step is destroyed
            del step

        # The single direction axis is squeezed by read_h5parm_gains
        dp3_gain, weight, solved, _ = read_h5parm_gains(h5parm_path)

    # Keep the initial values for terms DDECal did not solve
    _gain_gain = np.where(solved, dp3_gain, gain_gain).astype(gain_gain.dtype)
    _gain_weight = np.where(solved, weight, gain_weight).astype(
        gain_weight.dtype
    )

    return _gain_gain, _gain_weight, gain_residual.copy()


def _register_model_direction(
    dpinfo, phasecentre: tuple[float, float], tmpdir: str
):
    """
    Return a copy of ``dpinfo`` with a model direction registered, so
    that DDECal can reuse in-memory model data (``ddecal.reusemodel``).

    The DP3 Python bindings cannot add model directions to a DPInfo.
    DDECal with ``keepmodel`` registers its directions in its output
    DPInfo, so a DDECal step with a single-source sky model named
    ``_DP3_MODEL_DATA_NAME`` is only given the DPInfo. It does not process
    any data, so the sky model is never predicted.

    Parameters
    ----------
    dpinfo
        DPInfo describing the visibilities.
    phasecentre
        (ra, dec) in radians of the registered direction.
    tmpdir
        Directory for the temporary sky model.

    Returns
    -------
    dp3.DPInfo
        DPInfo with the direction ``_DP3_MODEL_STEP_PREFIX`` +
        ``_DP3_MODEL_DATA_NAME``.
    """
    import dp3  # pylint: disable=import-error,import-outside-toplevel
    from dp3.parameterset import (  # pylint: disable=import-error,import-outside-toplevel # noqa: E501
        ParameterSet,
    )

    # Plain floats, as repr() of numpy floats is not a valid angle
    ra, dec = (float(angle) for angle in phasecentre)
    skymodel_path = os.path.join(tmpdir, "model_direction.skymodel")
    with open(skymodel_path, "w", encoding="utf-8") as skymodel:
        skymodel.write(
            "FORMAT = Name, Type, Ra, Dec, I\n"
            f"{_DP3_MODEL_DATA_NAME}, POINT, {ra!r}rad, {dec!r}rad, 1.0\n"
        )

    prefix = _DP3_MODEL_STEP_PREFIX
    parset = ParameterSet()
    parset.add(f"{prefix}sourcedb", skymodel_path)
    parset.add(f"{prefix}onlypredict", "true")
    parset.add(f"{prefix}keepmodel", "true")
    step = dp3.make_step("ddecal", parset, prefix, dp3.MsType.regular)
    step.set_next_step(
        dp3.make_step("null", ParameterSet(), "", dp3.MsType.regular)
    )
    step.set_info(dpinfo)
    return step.info_out


def _split_skymodel_line(line: str) -> list[str]:
    """
    Split a sky model line on commas outside brackets, as values such as
    ``SpectralIndex`` are lists like ``[-0.7, 0.1]``. Empty fields are
    kept, so the fields line up with the format columns.
    """
    fields, current, depth = [], [], 0
    for char in line:
        if char == "," and depth == 0:
            fields.append("".join(current).strip())
            current = []
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        current.append(char)
    fields.append("".join(current).strip())
    return fields


def _parse_skymodel_angle(value: str) -> float:
    """
    Parse a sky model Ra or Dec value to radians.

    Uses casacore, like DP3, so ``197.9deg``, ``3.45rad``, ``13:11:39.5``
    and ``-22.16.40.7`` are all accepted.

    Raises
    ------
    ValueError
        If the value has no angle unit or format, e.g. ``1.2``, which can
        not be interpreted unambiguously.
    """
    # pylint: disable=import-error,import-outside-toplevel
    from casacore.quanta import quantity

    angle = quantity(value)
    if not angle.conforms(quantity("1rad")):
        raise ValueError(
            f"dp3_ddecal: sky model position {value!r} has no angle unit; "
            "use e.g. 'deg', or pass direction explicitly"
        )
    return angle.get_value("rad")


def _read_skymodel_patches(
    skymodel_path: str,
) -> dict[str, tuple[float, float] | None]:
    """
    Read the patches of a DP3 text sky model, with their directions.

    The directions are computed like DP3 does: a patch is at the
    flux-weighted mean direction of its sources, or at the position of its
    patch definition line if its sources have no flux. Sources without a
    patch are their own patch in DP3, named after the source.

    Parameters
    ----------
    skymodel_path
        Path to a DP3 text sky model.

    Returns
    -------
    dict
        Maps each patch name, in order of appearance, to its (ra, dec) in
        radians, or None if the patch has no position.

    Raises
    ------
    ValueError
        If the sky model is a sourcedb directory, has no format line or
        has a position without angle unit.
    """
    if os.path.isdir(skymodel_path):
        raise ValueError(
            "dp3_ddecal: direction must be given for a sourcedb directory "
            f"({skymodel_path})"
        )

    columns = None
    # Position given on the patch definition line
    patch_positions = {}
    # Flux-weighted sum of the source unit vectors, per patch
    patch_sums = {}
    with open(skymodel_path, encoding="utf-8") as skymodel:
        for line in skymodel:
            line = line.strip()
            if columns is None:
                # "FORMAT = Name, Type, ..." or "# (Name, Type, ...) = format"
                match = re.match(
                    r"^format\s*=\s*(.*)$|^#\s*\((.*)\)\s*=\s*format",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    header = match.group(1) or match.group(2)
                    # Column names may carry a default, e.g. "Patch='p1'"
                    columns = [
                        field.split("=")[0].strip().lower()
                        for field in _split_skymodel_line(header)
                    ]
                continue
            if not line or line.startswith("#"):
                continue

            fields = dict(zip(columns, _split_skymodel_line(line)))
            name = fields.get("name", "")
            patch = fields.get("patch", "") or name
            if not patch:
                continue
            position = None
            if fields.get("ra") and fields.get("dec"):
                position = (
                    _parse_skymodel_angle(fields["ra"]),
                    _parse_skymodel_angle(fields["dec"]),
                )

            if not name:
                # Patch definition line
                patch_positions[patch] = position
                patch_sums.setdefault(patch, np.zeros(4))
                continue
            sums = patch_sums.setdefault(patch, np.zeros(4))
            if position is not None:
                ra, dec = position
                flux = float(fields.get("i") or 0.0)
                # Unit vector of the source, weighted by its flux
                sums += flux * np.array(
                    [
                        np.cos(ra) * np.cos(dec),
                        np.sin(ra) * np.cos(dec),
                        np.sin(dec),
                        1.0,
                    ]
                )
                if patch not in patch_positions:
                    # Patch of a single source without a definition line
                    patch_positions[patch] = position

    if columns is None:
        raise ValueError(
            f"dp3_ddecal: no format line found in sky model {skymodel_path}"
        )

    patches = {}
    for patch, (x, y, z, flux) in patch_sums.items():
        if flux != 0:
            patches[patch] = (
                float(np.arctan2(y, x) % (2 * np.pi)),
                float(np.arctan2(z, np.hypot(x, y))),
            )
        else:
            patches[patch] = patch_positions.get(patch)
    return patches


def _find_phase_centre_patch(
    skymodel_path: str, phasecentre: tuple[float, float]
) -> str:
    """
    Find the sky model patch nearest to the phase centre.

    Parameters
    ----------
    skymodel_path
        Path to a DP3 text sky model.
    phasecentre
        Phase centre (ra, dec) in radians.

    Returns
    -------
    str
        Name of the patch whose direction is closest to the phase centre.

    Raises
    ------
    ValueError
        If no patch of the sky model has a position.
    """
    patches = {
        name: position
        for name, position in _read_skymodel_patches(skymodel_path).items()
        if position is not None
    }
    if not patches:
        raise ValueError(
            f"dp3_ddecal: no patch with a position in sky model "
            f"{skymodel_path}"
        )

    names = list(patches)
    ra, dec = np.array([patches[name] for name in names]).T
    ra0, dec0 = phasecentre
    # Angular distance to the phase centre (haversine formula)
    hav = (
        np.sin((dec - dec0) / 2) ** 2
        + np.cos(dec) * np.cos(dec0) * np.sin((ra - ra0) / 2) ** 2
    )
    separation = 2 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    nearest = int(np.argmin(separation))

    logger.info(
        "dp3_ddecal: solving direction independently for patch %s, "
        "%.1f arcsec from the phase centre",
        names[nearest],
        np.degrees(separation[nearest]) * 3600,
    )
    return names[nearest]
