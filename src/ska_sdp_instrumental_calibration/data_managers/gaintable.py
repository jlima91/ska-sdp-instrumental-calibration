from copy import copy
from typing import Literal, Union

import dask.array as da
import numpy as np
from ska_sdp_datamodels.science_data_model import ReceptorFrame
from ska_sdp_datamodels.visibility import Visibility

from .h5parm import read_h5parm_gains
from .schema.calibration_set import GainCalibrationSetXds
from .schema.polarization import jones_to_pol, pol_to_jones
from .solution_interval import SolutionIntervals


def create_gaintable_from_visibility(
    vis: Visibility,
    timeslice: Union[float, Literal["auto", "full"], None] = None,
    jones_type: Literal["T", "G", "B"] = "T",
    lower_precision: bool = True,
    skip_default_chunk: bool = False,
) -> GainCalibrationSetXds:
    """
    Create a unity- or identity-initialised GainTable consistent with the
    given Visibility.

    GainTable either represents:

    - a collection of complex-valued scalar gains, if Visibility carries only
      Stokes I data.
    - a collection of 2x2 complex-valued Jones matrices otherwise.

    In the first case, gains are initialised to unity. In the second case,
    Jones matrices are initialised to the identity matrix.

    Parameters
    ----------
    vis
        Visibility to create gaintable from
    timeslice
        Defines the time scale over which each gain solution is
        valid. This is used to define the time axis of the GainTable. This
        parameter is interpreted as follows depending on its type:

        - ``float``: this is a custom time interval in seconds.
          Input timestamps are grouped by intervals of this duration,
          and said groups are separately averaged to produce
          the output time axis.
        - "full": create a single solution across the entire visibility time
        - "auto" or ``None``: match the time resolution of the input
          visibility, i.e. copy the time axis of the input Visibility

    jones_type
        Type of Jones term, one of {"T", "G", "B"}.
        The frequency axis of the output GainTable depends on the value
        provided:

        - "B": the output frequency axis is the same as that of the input
          Visibility.
        - "T" or "G": solution is assumed to be frequency-independent,
          and the frequency axis of the output contains a single value: the
          average frequency of the input Visibility's channels.

    lower_precision
        Used to set up the float bit sizes while initialising the gaintable.
        If true, uses np.complex64 and np.float32 instead of higher precision
        np.complex128 and np.float64. Useful for memory optimization.
    skip_default_chunk
        If set to true, skips Dask/Xarray chunking of data in alignment to the
        input visibility. Useful in cases of chunk alignment issues.
        Default: False

    Returns
    -------
        Gaintable suitable for storing calibration solutions of given
        visibility
    """
    # Backward compatibility. Should be removed as "auto" is vary vague
    if timeslice == "auto":
        timeslice = None

    soln_intervals = SolutionIntervals(vis.time.data, timeslice)
    ntimes = soln_intervals.size

    nants = vis.visibility_acc.nants

    # Set the frequency sampling
    if jones_type == "B":
        gain_frequency = vis.frequency.data
        nfrequency = len(gain_frequency)
    elif jones_type in ("G", "T"):
        gain_frequency = np.mean(vis.frequency.data, keepdims=True)
        nfrequency = 1
    else:
        raise ValueError(f"Unknown Jones type {jones_type}")

    # There is only one receptor frame in Visibility
    # Use it for both receptor1 and receptor2
    receptor_frame = ReceptorFrame(vis.visibility_acc.polarisation_frame.type)
    nrec = receptor_frame.nrec
    npol = nrec * nrec

    gain_shape = [ntimes, nants, nfrequency, npol]

    # Create data variables with provided precision and backend
    if lower_precision:
        complex_dtype, float_dtype = np.complex64, np.float32
    else:
        complex_dtype, float_dtype = np.complex128, np.float64

    gain = da.broadcast_to(
        jones_to_pol(da.eye(nrec, dtype=complex_dtype)), gain_shape
    )
    gain_weight = da.ones(gain_shape, dtype=float_dtype)
    gain_residual = da.zeros([ntimes, nfrequency, npol], dtype=float_dtype)

    gain_table = GainCalibrationSetXds.constructor(
        gain=gain,
        time=soln_intervals.solution_time,
        interval=soln_intervals.intervals,
        weight=gain_weight,
        residual=gain_residual,
        frequency=gain_frequency,
        receptor_frame=receptor_frame,
        # phasecentre=vis.phasecentre,
        configuration=vis.configuration,
        jones_type=jones_type,
    )

    # Attach solution interval slices as attribute
    # pylint: disable-next=no-member
    gain_table.attrs["soln_interval_slices"] = (
        soln_intervals.indices
    )  # pylint: disable=E1101
    # Chunk data variables

    if skip_default_chunk:
        return gain_table

    gain_table = gain_table.chunk(time=1)  # pylint: disable=E1101
    if (
        gain_table.frequency.size == vis.frequency.size
        and "frequency" in vis.chunksizes
    ):
        gain_table = gain_table.chunk(frequency=vis.chunksizes["frequency"])

    return gain_table


def reset_gaintable(gaintable: GainCalibrationSetXds) -> GainCalibrationSetXds:
    """
    Returns a new dask-backed gaintable with all data variables resetted
    to their initial sensible values.

    Parameters
    ----------
    gaintable
        Gaintable object to be reset

    Returns
    -------
        Gaintable with data variables resetted to their intial sensible values.
    """
    gain_shape = gaintable.CALPARAM_GAIN.shape
    nrec = gaintable.calibration_set.nrec
    gain = da.broadcast_to(
        jones_to_pol(da.eye(nrec, dtype=gaintable.CALPARAM_GAIN.dtype)),
        gain_shape,
    )

    weight = da.ones(
        gaintable.CALPARAM_WEIGHT.shape, dtype=gaintable.CALPARAM_WEIGHT.dtype
    )

    residual = da.zeros(
        gaintable.CALPARAM_RESIDUAL.shape,
        dtype=gaintable.CALPARAM_RESIDUAL.dtype,
    )

    # Deepcopy and change data variables
    # Simpler and less prone to errors than "assign"
    new_gaintable = gaintable.copy(deep=True)
    new_gaintable.CALPARAM_GAIN.data = gain
    new_gaintable.CALPARAM_WEIGHT.data = weight
    new_gaintable.CALPARAM_RESIDUAL.data = residual

    return new_gaintable


def divide_bandpass_by_ref_ant_preserve_phase(
    gaintable: GainCalibrationSetXds, ref_ant: int
) -> GainCalibrationSetXds:
    """
    Original code referred from https://github.com/flint-crew/flint.git
    Divide the bandpass complex gains (solved for initially by something
    like calibrate) by a nominated reference antenna. In the case of
    ``calibrate`` there is no implicit reference antenna. This is valid for
    cases where the xy-phase is set to 0 (true via the ASKAP on-dish
    calibrator).

    This particular function is most appropriate for the `calibrate` style
    solutions, which solve for the Jones in one step. In HMS notation this
    are normally split into two separate 2x2 matrices, one for the gains
    with zero off-diagonal elements and a leakage matrix with ones on
    the diagonal.

    This is the preferred function to use whena attempting to set a
    phase reference antenna to precomputed Jones bandpass solutions.

    The input complex gains should be in the form:
    >> (ant, channel, pol)

    Internally reference phasores are constructed for the G_x and G_y
    terms of the reference antenna. They are then applied:
    >> G_x = G_x / G_xref
    >> G_xyp = G_xy / G_yref
    >> G_yxp = G_yx / G_xref
    >> G_y = G_y / G_yref

    which is applied to all antennas in ``complex_gains``.

    Args:
        complex_gains (np.ndarray): The complex gains that will be normalised
        ref_ant (int): The desired reference antenna to use

    Returns:
        GainTable: The normalised bandpass solutions
    """
    gains = pol_to_jones(np.copy(gaintable.CALPARAM_GAIN.data))

    g_x = gains[..., 0, 0]
    g_y = gains[..., 1, 1]
    g_xy = gains[..., 0, 1]
    g_yx = gains[..., 1, 0]

    ref_g_x = gains[:, ref_ant : ref_ant + 1, :, 0, 0]  # noqa: E203
    ref_g_x = ref_g_x / np.abs(ref_g_x)

    ref_g_y = gains[:, ref_ant : ref_ant + 1, :, 1, 1]  # noqa: E203
    ref_g_y = ref_g_y / np.abs(ref_g_y)

    g_x_prime = g_x / ref_g_x
    g_y_prime = g_y / ref_g_y
    g_xy_prime = g_xy / ref_g_y
    g_yx_prime = g_yx / ref_g_x

    bp_p = np.zeros_like(gains) * np.nan + 1j * np.zeros_like(gains) * np.nan

    bp_p[..., 0, 0] = g_x_prime
    bp_p[..., 1, 1] = g_y_prime
    bp_p[..., 0, 1] = g_xy_prime
    bp_p[..., 1, 0] = g_yx_prime

    new_gain = gaintable.CALPARAM_GAIN.copy()
    new_gain.data = jones_to_pol(bp_p)

    return gaintable.assign(
        {
            "CALPARAM_GAIN": new_gain,
        }
    ).chunk(gaintable.chunks)


def multiply_gaintables(
    gt: GainCalibrationSetXds, dgt: GainCalibrationSetXds, time_tolerance=1e-3
) -> GainCalibrationSetXds:
    """
    Multiply two GainCalibrationSetXds.

    Returns gt * dgt.

    :param gt: First GainCalibrationSetXds
    :param dgt: Second GainCalibrationSetXds
    :param time_tolerance: Maximum tolerance of time
                separation in the GainCalibrationSetXds data
    :return: Multiplication product
    """

    # Test if times align
    mismatch = np.max(np.abs(gt["time"].data - dgt["time"].data))
    if mismatch > time_tolerance:
        raise ValueError(
            f"Gaintables not aligned in time: max mismatch {mismatch} seconds"
        )
    if dgt.calibration_set.nrec == gt.calibration_set.nrec:
        if dgt.calibration_set.nrec == 2:
            gt["CALPARAM_GAIN"].data = jones_to_pol(
                np.einsum(
                    "...ki,...ij->...kj",
                    pol_to_jones(gt["CALPARAM_GAIN"].data),
                    pol_to_jones(dgt["CALPARAM_GAIN"].data),
                )
            )
            gt["CALPARAM_WEIGHT"].data *= dgt["CALPARAM_WEIGHT"].data
        elif dgt.calibration_set.nrec == 1:
            gt["CALPARAM_GAIN"].data *= dgt["CALPARAM_GAIN"].data
            gt["CALPARAM_WEIGHT"].data *= dgt["CALPARAM_WEIGHT"].data
        else:
            raise ValueError(
                f"Gain tables have illegal structures {str(gt)} {str(dgt)}"
            )

    else:
        raise ValueError(
            f"Gain tables have different structures {str(gt)} {str(dgt)}"
        )

    return gt


def _check_antennas(
    h5parm_antennas: list[str] | None, vis_antennas: list[str]
) -> None:
    """
    Check that the h5parm antennas match the visibility antennas.

    Parameters
    ----------
    h5parm_antennas
        Antenna names along the antenna axis of the h5parm, or None if the
        antenna axis is absent in the h5parm.
    vis_antennas
        Antenna names of the visibility configuration.

    Raises
    ------
    ValueError
        If the antennas or their order differ, or the h5parm has no
        antenna axis while the visibility has more than one antenna.
    """
    if h5parm_antennas is None:
        if len(vis_antennas) != 1:
            raise ValueError(
                "h5parm has no antenna axis, but visibility has "
                f"{len(vis_antennas)} antennas"
            )
        return

    if h5parm_antennas != vis_antennas:
        raise ValueError(
            "h5parm antennas do not match the visibility antennas "
            f"(in order). h5parm: {h5parm_antennas}, "
            f"visibility: {vis_antennas}"
        )


def create_gaintable_from_h5parm(
    h5parm_path: str,
    interval: np.ndarray,
    vis: Visibility | None = None,
    jones_type: Literal["T", "G", "B"] = "T",
) -> GainCalibrationSetXds:
    """
    Load an H5Parm file and convert it to a GainCalibrationSetXds.

    Supports H5Parm files written by INST and DP3 (gaincal).

    Data axes are reordered to (time, antenna, frequency, receptor1,
    receptor2) based on the AXES attribute. The receptor dimensions
    are then flattened into the polarization dimension required by
    GainCalibrationSetXds.

    If only diagonal polarisations are present, cross-polarization
    terms are set to zero. If either the amplitude or phase soltab
    is absent, unit amplitude or zero phase is assumed. If both are
    present, the minimum of their weights is used.

    If vis is provided, the antennas in the H5Parm file must match
    the visibility configuration in the same order. Missing time or
    frequency axes are restored using the mean values from vis.

    Parameters
    ----------
    h5parm_path
        Path to the H5Parm file.
    interval
        Interval array for the gain table, typically derived from
        SolutionIntervals.intervals.
    vis
        Visibility used to obtain metadata such as configuration and
        receptor frame. Required if the H5Parm has no time or
        frequency axis.
    jones_type
        Jones term type: "T", "G", or "B".

    Returns
    -------
    GainCalibrationSetXds
        Calibration solutions represented using the new schema.
    """
    gain, weight, _, soltab = read_h5parm_gains(h5parm_path)

    time = soltab.time
    frequency = soltab.frequency

    if vis is None and (time is None or frequency is None):
        raise ValueError(
            f"H5Parm {h5parm_path} has no time or frequency axis; "
            "vis is required to restore them"
        )

    if time is None:
        time = np.mean(vis.time.data, keepdims=True)

    if frequency is None:
        frequency = np.mean(vis.frequency.data, keepdims=True)

    if vis is not None:
        _check_antennas(
            soltab.antenna,
            [str(name) for name in vis.configuration.names.data],
        )

    gain = jones_to_pol(gain)
    weight = jones_to_pol(weight)

    kwargs = {}

    if vis is not None:
        kwargs["configuration"] = copy(vis.configuration)
        kwargs["receptor_frame"] = ReceptorFrame(
            vis.visibility_acc.polarisation_frame.type
        )

    residual = np.zeros(
        (len(time), len(frequency), gain.shape[-1]),
        dtype=np.float32,
    )

    gaintable = GainCalibrationSetXds.constructor(
        gain=gain,
        time=time,
        interval=interval,
        weight=weight,
        residual=residual,
        frequency=frequency,
        jones_type=jones_type,
        **kwargs,
    )

    return gaintable
