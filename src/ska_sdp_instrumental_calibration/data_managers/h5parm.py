"""
Helpers to read and write solution tables (soltabs) of H5Parm files.

H5Parm is the HDF5 schema used by LOFAR software (losoto, DP3) to store
calibration solutions. Solutions are stored in solution sets
(``solset``, e.g. ``sol000``) which contain solution tables
(``soltab``, e.g. ``amplitude000``). Each soltab contains a ``val`` and
a ``weight`` dataset, whose axis order is given by their ``AXES``
attribute, and one dataset per axis holding the axis coordinates.
"""

from dataclasses import dataclass
from typing import Iterable, Literal

import h5py
import numpy as np
from numpy.typing import NDArray
from ska_sdp_datamodels.calibration.calibration_model import GainTable

from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable

__all__ = [
    "SoltabData",
    "create_soltab_group",
    "create_soltab_datasets",
    "create_clock_soltab_datasets",
    "to_null_terminated_bytes",
    "read_soltab",
    "combine_amplitude_phase",
    "read_h5parm_gains",
]

_SOLTAB_AXES = ("time", "antenna", "frequency")

_H5PARM_AXIS_ALIASES = {
    "time": "time",
    "ant": "antenna",
    "antenna": "antenna",
    "freq": "frequency",
    "frequency": "frequency",
    "pol": "pol",
}

_POL_TO_RECEPTOR_INDEX = {
    "XX": (0, 0),
    "XY": (0, 1),
    "YX": (1, 0),
    "YY": (1, 1),
    "RR": (0, 0),
    "RL": (0, 1),
    "LR": (1, 0),
    "LL": (1, 1),
}


@dataclass
class SoltabData:
    """
    Contents of a soltab, re-ordered into Jones-matrix shaped arrays.

    Parameters
    ----------
    values
        Solution values. Shape: (time, antenna, frequency, 2, 2).
    weights
        Solution weights. Shape matches ``values``.
    solved
        Boolean mask of the receptor pairs present in the soltab.
        Shape matches ``values``.
    time
        Time coordinates, or None if the time axis is absent in the
        soltab (e.g. squeezed on export).
    frequency
        Frequency coordinates, or None if the frequency axis is absent.
    antenna
        Antenna names, or None if the antenna axis is absent.
    """

    values: np.ndarray
    weights: np.ndarray
    solved: np.ndarray
    time: np.ndarray | None
    frequency: np.ndarray | None
    antenna: list[str] | None


def create_soltab_group(
    solset: h5py.Group, solution_type: Literal["amplitude", "phase", "clock"]
) -> h5py.Group:
    """Create soltab group under given solset group.

    :param solset: base-level HDF5 group to update
    :param solution_type: only "amplitude" and "phase" are supported at present
    :return: HDF5 group for the "solution_type" data
    """
    soltab = solset.create_group(f"{solution_type}000")
    soltab.attrs["TITLE"] = np.bytes_(solution_type)
    return soltab


def create_soltab_datasets(soltab: h5py.Group, gaintable: GainTable):
    """Add a dataset for each of the GainTable dimensions.

    :param soltab: HDF5 table to update
    :param gaintable: GainTable
    """
    # create a dataset for each dimension
    for dim in list(gaintable.gain.sizes):
        soltab.create_dataset(dim, data=gaintable[dim].data)

    # create datasets for the data and weights
    shape = gaintable.gain.shape
    axes = np.bytes_(",".join(list(gaintable.gain.sizes)))

    val = soltab.create_dataset("val", shape=shape, dtype=float)
    val.attrs["AXES"] = axes

    weight = soltab.create_dataset("weight", shape=shape, dtype=float)
    weight.attrs["AXES"] = axes

    return val, weight


def create_clock_soltab_datasets(soltab: h5py.Group, delaytable: DelayTable):
    """Add a dataset for each of the Delay dimensions.

    :param soltab: HDF5 table to update
    :param delaytable: xr.Dataset
    """
    # create a dataset for each dimension
    for dim in list(delaytable.delay.sizes):
        soltab.create_dataset(dim, data=delaytable[dim].data)

    # create datasets for the data and weights
    shape = delaytable.delay.shape
    axes = np.bytes_(",".join(list(delaytable.delay.sizes)))

    val = soltab.create_dataset("val", shape=shape, dtype=float)
    val.attrs["AXES"] = axes

    offset = soltab.create_dataset("offset", shape=shape, dtype=float)
    offset.attrs["AXES"] = axes

    return val, offset


def to_null_terminated_bytes(strings: Iterable[str]) -> NDArray:
    """
    Encode strings as null-terminated ASCII bytes.

    Parameters
    ----------
    strings
        Strings to encode.

    Returns
    -------
    NDArray
        Array of null-terminated byte strings.
    """
    # NOTE: making antenna names one character longer, in keeping with
    # ska-sdp-batch-preprocess
    return np.asarray([s.encode("ascii") + b"\0" for s in strings])


def _decode_strings(values: np.ndarray) -> list[str]:
    """
    Decode (possibly null-terminated) byte strings of an h5parm dataset.

    Parameters
    ----------
    values
        Byte strings or strings.

    Returns
    -------
    list of str
        Decoded strings without trailing null characters.
    """
    return [
        (v.decode() if isinstance(v, bytes) else str(v)).strip("\x00")
        for v in values
    ]


def read_soltab(soltab: h5py.Group) -> SoltabData:
    """
    Read values and weights of a soltab in (time, antenna, frequency, 2, 2).

    Axis order is taken from the ``AXES`` attribute of the ``val`` dataset,
    so that both INST (``time,ant,freq,pol``) and DP3
    (``time,freq,ant,pol``) layouts are supported. Absent time, antenna or
    frequency axes (e.g. squeezed on export) are restored with length one,
    and their coordinates are set to None. If only diagonal polarisations
    are present, cross terms are filled with zeros (with zero weight). If
    the polarisation axis is absent (scalar solutions), the value is used
    for both diagonal terms. If the ``weight`` dataset is absent, unit
    weights are used.

    Parameters
    ----------
    soltab
        h5parm soltab group, e.g. ``sol000/amplitude000``.

    Returns
    -------
    SoltabData
        Re-ordered soltab values, weights and coordinates.

    Raises
    ------
    ValueError
        If the soltab has unsupported axes or polarisations.
    """
    raw_axes = soltab["val"].attrs["AXES"]
    if isinstance(raw_axes, bytes):
        raw_axes = raw_axes.decode()
    raw_axes = str(raw_axes).split(",")
    axes = [_H5PARM_AXIS_ALIASES.get(ax, ax) for ax in raw_axes]
    # Map normalised axis name -> dataset name inside the soltab
    h5_axes = dict(zip(axes, raw_axes))

    unknown = set(axes) - {*_SOLTAB_AXES, "pol"}
    if unknown:
        raise ValueError(
            f"h5parm soltab {soltab.name} has unsupported axes "
            f"{sorted(unknown)}"
        )

    val = soltab["val"][...]
    weight = soltab["weight"][...] if "weight" in soltab else np.ones_like(val)

    # Restore absent axes with length one
    for ax in _SOLTAB_AXES:
        if ax not in axes:
            axes.append(ax)
            val = val[..., np.newaxis]
            weight = weight[..., np.newaxis]

    has_pol = "pol" in axes
    order = [*_SOLTAB_AXES] + (["pol"] if has_pol else [])
    perm = [axes.index(ax) for ax in order]
    val = np.transpose(val, perm)
    weight = np.transpose(weight, perm)

    out_shape = (*val.shape[:3], 2, 2)
    out_val = np.zeros(out_shape, dtype=val.dtype)
    out_weight = np.zeros(out_shape, dtype=weight.dtype)
    solved = np.zeros((2, 2), dtype=bool)

    if has_pol:
        pols = _decode_strings(soltab[h5_axes["pol"]][...])
        for idx, pol in enumerate(pols):
            if pol not in _POL_TO_RECEPTOR_INDEX:
                raise ValueError(f"Unsupported polarisation {pol!r}")
            r1, r2 = _POL_TO_RECEPTOR_INDEX[pol]
            out_val[..., r1, r2] = val[..., idx]
            out_weight[..., r1, r2] = weight[..., idx]
            solved[r1, r2] = True
    else:
        # scalar solutions apply equally to both receptors
        for r in range(2):
            out_val[..., r, r] = val
            out_weight[..., r, r] = weight
            solved[r, r] = True

    def _coords(ax: str) -> np.ndarray | None:
        return soltab[h5_axes[ax]][...] if ax in h5_axes else None

    antenna = _coords("antenna")

    return SoltabData(
        values=out_val,
        weights=out_weight,
        solved=np.broadcast_to(solved, out_shape),
        time=_coords("time"),
        frequency=_coords("frequency"),
        antenna=None if antenna is None else _decode_strings(antenna),
    )


def combine_amplitude_phase(
    amplitude: SoltabData | None, phase: SoltabData | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine amplitude and phase soltabs into complex gains.

    If both soltabs are given, the minimum of their weights is used. If
    only one is given, unit amplitude (on the diagonal terms) or zero
    phase is assumed for the other.

    Parameters
    ----------
    amplitude
        Amplitude soltab, or None if absent.
    phase
        Phase soltab, or None if absent.

    Returns
    -------
    tuple of np.ndarray
        ``(gain, weight, solved)`` each of shape
        (time, antenna, frequency, 2, 2).

    Raises
    ------
    ValueError
        If both soltabs are None.
    """
    if amplitude is None and phase is None:
        raise ValueError("Either amplitude or phase soltab is required")

    if amplitude is not None and phase is not None:
        amp_val = amplitude.values
        phase_val = phase.values
        weight = np.minimum(amplitude.weights, phase.weights)
        solved = amplitude.solved & phase.solved
    elif amplitude is not None:
        amp_val = amplitude.values
        phase_val = np.zeros_like(amp_val)
        weight = amplitude.weights
        solved = amplitude.solved
    else:
        phase_val = phase.values
        # Unit amplitude on diagonal terms, cross terms stay zero
        amp_val = np.zeros_like(phase_val)
        amp_val[..., 0, 0] = amp_val[..., 1, 1] = 1.0
        weight = phase.weights
        solved = phase.solved

    return amp_val * np.exp(1j * phase_val), weight, solved


def read_h5parm_gains(
    h5parm_path: str, solset: str = "sol000"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SoltabData]:
    """
    Read complex gains from the amplitude and phase soltabs of an h5parm.

    Supports h5parm files written by INST as well as by DP3 (gaincal).
    If either the ``amplitude000`` or ``phase000`` soltab is absent
    (e.g. DP3 ``caltype=diagonalphase``), unit amplitude or zero phase
    is assumed.

    Parameters
    ----------
    h5parm_path
        Path to the h5parm file.
    solset
        Name of the solset to read.

    Returns
    -------
    tuple
        ``(gain, weight, solved, soltab)`` where ``gain``, ``weight`` and
        ``solved`` have shape (time, antenna, frequency, 2, 2), and
        ``soltab`` is the amplitude soltab (or the phase soltab if
        amplitude is absent), providing the axis coordinates.

    Raises
    ------
    ValueError
        If neither ``amplitude000`` nor ``phase000`` soltab is present.
    """
    with h5py.File(h5parm_path, "r") as h5f:
        solution = h5f[solset]
        amplitude = (
            read_soltab(solution["amplitude000"])
            if "amplitude000" in solution
            else None
        )
        phase = (
            read_soltab(solution["phase000"])
            if "phase000" in solution
            else None
        )

    if amplitude is None and phase is None:
        raise ValueError(
            f"No amplitude000 or phase000 soltab found in "
            f"{h5parm_path}:{solset}"
        )

    gain, weight, solved = combine_amplitude_phase(amplitude, phase)
    return gain, weight, solved, amplitude or phase
