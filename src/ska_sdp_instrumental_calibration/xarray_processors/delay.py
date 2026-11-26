import logging
import warnings
from functools import partial

import numpy as np
import xarray as xr
from numpy.exceptions import ComplexWarning

from ..data_managers.processing_set import (
    get_baseline_antenna_indices,
    get_correlated_data,
    get_data_group,
)
from ..data_managers.schema.calibration_set import GainCalibrationSetXds
from ._utils import with_chunks

logger = logging.getLogger()


class DelayTable(xr.Dataset):
    """
    Container for per-antenna, per-polarisation delay solutions computed
    from a GainCalibrationSetXds.

    **Coordinates**

    - time: time centroids of solutions, in seconds elapsed since the MJD
      reference epoch, ``[ntimes]``.

    - antenna_name: antenna names, ``[nants]``.

    - pol: polarisation labels, e.g. ``['XX', 'YY']``, ``[npol]``.

    **Data variables**

    - delay: per-antenna, per-polarisation group-delay in seconds,
      real-valued ``[ntimes, nants, npol]``.

    - offset: per-antenna, per-polarisation phase offset in cycles,
      real-valued ``[ntimes, nants, npol]``.

    **Attributes**

    - data_model: name of this class, used internally for saving to / loading
      from files.

    Here is an example::

        <xarray.DelayTable>
        Dimensions:       (time: 1, antenna_name: 115, pol: 2)
        Coordinates:
          * time          (time) float64 5.085e+09
          * antenna_name  (antenna_name) <U7 'C1' 'C2' ... 'S16-6'
          * pol           (pol) <U2 'XX' 'YY'
        Data variables:
            delay         (time, antenna_name, pol) float64 ...
            offset        (time, antenna_name, pol) float64 ...
        Attributes:
            data_model:     DelayTable
    """

    __slots__ = ()

    def __init__(self, data_vars=None, coords=None, attrs=None):
        super().__init__(data_vars, coords=coords, attrs=attrs)

    @classmethod
    def constructor(
        cls,
        delay: np.ndarray,
        offset: np.ndarray,
        time: np.ndarray,
        antenna_name: np.ndarray,
        pol: list[str],
    ) -> "DelayTable":
        """
        Create a DelayTable instance directly from numpy arrays.

        Parameters
        ----------
        delay
            Per-antenna, per-polarisation group-delay in seconds
            ``[ntimes, nants, npol]``.
        offset
            Per-antenna, per-polarisation phase offset in cycles
            ``[ntimes, nants, npol]``.
        time
            Centroids of solutions, in seconds elapsed since the MJD
            reference epoch ``[ntimes]``.
        antenna_name
            Antenna names ``[nants]``.
        pol
            Polarisation labels, e.g. ``['XX', 'YY']``.
        """
        coords = {
            "time": time,
            "antenna_name": antenna_name,
            "pol": list(pol),
        }
        dims = ["time", "antenna_name", "pol"]
        datavars = {
            "delay": xr.DataArray(delay, dims=dims),
            "offset": xr.DataArray(offset, dims=dims),
        }
        attrs = {"data_model": "DelayTable"}
        return cls(data_vars=datavars, coords=coords, attrs=attrs)


def calculate_delays_from_gain(
    gaintable: GainCalibrationSetXds, oversample: int
) -> DelayTable:
    """
    Calculates per-antenna delays and phase offsets from the gains of
    the given gaintable

    Parameters
    ----------
    gaintable
        GainCalibrationSetXds
    oversample
        Oversample rate required for the delay

    Returns
    -------
        A dataset holding delay data
    """
    # Delay fit needs the whole band, so only frequency is rechunked.
    # Time and antenna keep the gaintable chunks, so that delays for
    # all antennas in a chunk are calculated in a single task.
    gain = gaintable["CALPARAM_GAIN"].chunk(frequency=-1)
    weight = gaintable["CALPARAM_WEIGHT"].chunk(frequency=-1)
    # We calculate delays only for the diagonal terms
    pols = gaintable["polarization"].data[[0, 3]].tolist()

    with warnings.catch_warnings():
        # apply_ufunc throws a false warning, when it detects that
        # one of the inputs (gain) has complex dtype, but outputs
        # are all non-complex values
        warnings.simplefilter("ignore", ComplexWarning)

        delay, offset = xr.apply_ufunc(
            _calculate_delays_ufunc_,
            gain,
            weight,
            input_core_dims=[
                ["frequency", "polarization"],
                ["frequency", "polarization"],
            ],
            output_core_dims=[["pol"], ["pol"]],
            output_dtypes=[np.float64, np.float64],
            dask="parallelized",
            dask_gufunc_kwargs=dict(output_sizes={"pol": len(pols)}),
            kwargs=dict(
                frequency=gaintable["frequency"].data,
                oversample=oversample,
            ),
        )

    return DelayTable.constructor(
        delay=delay.transpose("time", "antenna_name", "pol").data,
        offset=offset.transpose("time", "antenna_name", "pol").data,
        time=gaintable["time"].values,
        antenna_name=gaintable["antenna_name"].values,
        pol=pols,
    )


def _calculate_delays_ufunc_(
    gain: np.ndarray,
    weight: np.ndarray,
    frequency: np.ndarray,
    oversample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function acts as a bridge between
    calculate_delays_from_gain and numpy based
    delay functions (coarse_delay, update_delay)

    Delays are calculated at once for all times and antennas
    of the chunk, using the full frequency band, and only for
    the diagonal terms (XX, YY).

    Parameters
    ----------
    gain: (time, nant, freq, 4) (np.complex64)
    weight: (time, nant, freq, 4) (float)
    frequency: (freq,) (float)
    oversample: int

    Returns
    -------
        delay: (time, nant, 2)
        offset: (time, nant, 2)
    """
    # Select only diagonal terms
    # Keeping frequency as the last axis
    # Shape: (time, nant, pol, freq)
    gain_diagonal = gain[..., [0, 3]].transpose(0, 1, 3, 2)
    weight_diagonal = weight[..., [0, 3]].transpose(0, 1, 3, 2)

    delay_coarse = coarse_delay(gain_diagonal, frequency, oversample)
    delay, offset = update_delay(
        gain_diagonal,
        weight_diagonal,
        frequency,
        np.zeros_like(delay_coarse),
        delay_coarse,
    )

    # Remove the trailing size 1 axis (frequency)
    # left by the numpy functions
    return delay[..., 0], offset[..., 0]


def apply_delay_to_gaintable(
    gaintable: GainCalibrationSetXds,
    delaytable: DelayTable,
    inverse: bool = False,
) -> GainCalibrationSetXds:
    """
    Applies the delay to the given gaintable

    Parameters
    ----------
    gaintable
        Gaintable on which we need to apply delays
    delaytable
        DelayTable which holds calculated delays
    inverse
        Whether to invert the delays before applying

    Returns
    -------
        Gaintable with updated gains
    """
    # Gains are merged across frequency, so that the whole band is
    # processed in a single task, instead of a task per frequency chunk.
    gain = gaintable["CALPARAM_GAIN"].chunk(frequency=-1)

    # Frequency is passed as an argument (not in kwargs), so that
    # xarray aligns it with the frequency dimension of the gains.
    delay_rotated_gain = xr.apply_ufunc(
        _apply_delay_ufunc_,
        gain,
        delaytable["delay"],
        delaytable["offset"],
        gaintable["frequency"],
        input_core_dims=[["polarization"], ["pol"], ["pol"], []],
        output_core_dims=[["polarization"]],
        output_dtypes=[gaintable["CALPARAM_GAIN"].dtype],
        dask="parallelized",
        kwargs=dict(inverse=inverse),
    )

    # Restore the chunks of the input gaintable
    return gaintable.assign(
        CALPARAM_GAIN=with_chunks(
            delay_rotated_gain.transpose(*gaintable["CALPARAM_GAIN"].dims),
            gaintable["CALPARAM_GAIN"].chunksizes,
        )
    )


def _apply_delay_ufunc_(
    gain: np.ndarray,
    delay: np.ndarray,
    offset: np.ndarray,
    frequency: np.ndarray,
    inverse: bool,
) -> np.ndarray:
    """
    This function acts as a bridge between
    apply_delay_to_gaintable and calculate_gain_rot

    Delays are applied only to the diagonal terms of
    the 2x2 gain matrix. Off-diagonal terms are set to zero.

    Parameters
    ----------
    gain: (time, nant, freq, 4) (complex)
    delay: (time, nant, 1, 2) (float)
        apply_ufunc inserts a size 1 axis for frequency
    offset: (time, nant, 1, 2) (float)
    frequency: (freq,) (float)
    inverse: bool
        Whether to invert the delays before applying

    Returns
    -------
        Gains with delays applied, same shape and dtype as gain
    """
    # Select only diagonal terms
    gain_diagonal = gain[..., [0, 3]]

    rotated_diagonal = calculate_gain_rot(
        gain_diagonal,
        delay,
        offset,
        frequency[..., np.newaxis],
        inverse=inverse,
    )

    rotated_gain = np.zeros_like(gain)
    rotated_gain[..., 0] = rotated_diagonal[..., 0]
    rotated_gain[..., 3] = rotated_diagonal[..., 1]

    return rotated_gain


def update_delay(
    gains: np.ndarray,
    wgt: np.ndarray,
    freq: np.ndarray,
    _offset: np.ndarray,
    delay: np.ndarray,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Refines the delay and offset by a weighted least-squares linear fit
    of the residual phase (in cycles) against frequency

    Parameters
    ----------
    gains: np.ndarray. complex. (..., freq)
    wgt: np.ndarray. float (..., freq)
    freq: np.ndarray. float (freq,)
    _offset: np.ndarray. float (..., 1)
        Initial offset per station
    delay: np.ndarray. float (..., 1)
        Initial delay per station

    Returns
    -------
        Updated delay and offset
        Each a np.ndarray, Shape: (..., 1)
    """
    gains_rot = calculate_gain_rot(gains, delay, _offset, freq, inverse=True)

    cycles = np.unwrap(np.angle(gains_rot)) / (2 * np.pi)

    # Sum across frequency, keeping the axis for broadcasting
    sum_freq = partial(np.sum, axis=-1, keepdims=True)

    denom = sum_freq(wgt) * sum_freq(wgt * freq * freq) - sum_freq(
        wgt * freq
    ) * sum_freq(wgt * freq)

    offset = (
        _offset
        + (
            sum_freq(wgt * freq * freq) * sum_freq(wgt * cycles)
            - sum_freq(wgt * freq) * sum_freq(wgt * freq * cycles)
        )
        / denom
    )

    updated_delay = (
        sum_freq(wgt) * sum_freq(wgt * cycles * freq)
        - sum_freq(wgt * cycles) * sum_freq(wgt * freq)
    ) / denom

    return delay + updated_delay, offset


def coarse_delay(
    gains: np.ndarray, frequency: np.ndarray, oversample: int
) -> np.ndarray[float]:
    """
    Calculates the coarse delay

    Parameters
    ----------
    gains: np.ndarray. complex. (..., freq)
    frequency: np.ndarray. float (freq,)
    oversample: int
        Oversample rate

    Returns
    -------
        Delay value for given frequency range
        np.ndarray, Shape (..., 1)
    """
    nchan = frequency.size
    N = oversample * nchan
    gain_start_index = N // 2 - nchan // 2
    gain_stop_index = gain_start_index + nchan

    padded_gains = np.zeros(gains.shape[:-1] + (N,), gains.dtype)
    padded_gains[..., gain_start_index:gain_stop_index] = gains

    delay_spec = np.fft.fftshift(np.fft.fft(padded_gains, axis=-1), axes=-1)

    delay = ((1 / oversample) / (frequency[-1] - frequency[0])) * (
        -N // 2 + np.arange(N)
    )

    return delay[np.abs(delay_spec).argmax(axis=-1, keepdims=True)]


def calculate_gain_rot(
    gain: np.ndarray,
    delay: np.ndarray | float,
    offset: np.ndarray | float,
    frequency: np.ndarray,
    inverse=False,
) -> np.ndarray:
    """
    Calculates gain rotation.
    The function assumes that for numpy arrays as input, the values
    are broadcastable across all other dimensions
    excluding the "freq" dimension.

    Parameters
    ----------
    gain
        Shape: (..., freq,) Dtype: complex64
    delay
        Shape: (..., 1) Dtype: float
    offset
        Shape: (..., 1) Dtype: float
    frequency
        Shape: (freq,) Dtype: float
    inverse: bool
        Whether to invert the phases before applying

    Returns
    -------
        Array of calculated gain rotation
        Same shape and dtype as gain
    """
    sign = -1 if inverse else 1
    return gain * np.exp(sign * 2j * np.pi * (offset + (delay * frequency)))


def _refant_baselines(
    ms_xdt: xr.DataTree, refant: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each antenna, the baseline it forms with the reference antenna.

    Parameters
    ----------
    ms_xdt
        MSv4 node.
    refant
        Index of the reference antenna, in ``antenna_xds`` order.

    Returns
    -------
        For each antenna in ``antenna_xds`` order: the index of its baseline
        with the reference antenna (0 if there is none), whether that
        baseline is stored as (refant, antenna) and must be conjugated to
        give V(antenna, refant), and whether there is no such baseline. The
        reference antenna itself is marked as having no baseline.

    Raises
    ------
    ValueError
        If an antenna has more than one baseline with the reference antenna.
    """
    antenna1, antenna2 = get_baseline_antenna_indices(ms_xdt)
    nants = ms_xdt["antenna_xds"].ds.antenna_name.size

    baseline_index = np.zeros(nants, dtype=np.int64)
    conjugate = np.zeros(nants, dtype=bool)
    missing = np.ones(nants, dtype=bool)
    for baseline, (a1, a2) in enumerate(zip(antenna1, antenna2)):
        if a1 == a2 or refant not in (a1, a2):
            continue
        antenna = a1 if a2 == refant else a2
        if not missing[antenna]:
            raise ValueError(
                f"Antenna {antenna} has several baselines with the "
                f"reference antenna {refant}"
            )
        baseline_index[antenna] = baseline
        conjugate[antenna] = a1 == refant
        missing[antenna] = False

    return baseline_index, conjugate, missing


def create_delaytable_from_vis(
    vis: xr.DataTree,
    gaintable: GainCalibrationSetXds,
    refant: int,
    oversample: int,
) -> DelayTable:
    """
    Calculates delays from visibility data by processing each solution interval

    For each antenna, the delays are fitted to the time averaged
    visibilities of its baseline with the reference antenna, as
    V(antenna, refant), with their weights. The reference antenna is given
    a unit gain, and antennas without a baseline with the reference antenna
    are given zero visibilities and weights.

    Parameters
    ----------
    vis
        MSv4 node holding a single data group. If backed by a dask array,
        can be chunked in time and frequency axis.
    gaintable
        Gaintable containing solution intervals, created from ``vis``.
    refant
        Index of the reference antenna, in ``antenna_xds`` order.
    oversample
        Oversample rate required for the delay

    Returns
    -------
        Dataset of calculated delays
    """
    baseline_index, conjugate, missing = _refant_baselines(vis, refant)
    is_refant = np.arange(baseline_index.size) == refant

    def per_antenna(mask: np.ndarray) -> xr.DataArray:
        return xr.DataArray(mask, dims="antenna_name")

    correlated_data = get_correlated_data(vis)
    weight = vis.ds[get_data_group(vis)["weight"]]

    gaintable = gaintable.rename(time="solution_time")

    soln_interval_slices = gaintable.soln_interval_slices

    delay_table_across_solutions = []

    for idx, slc in enumerate(soln_interval_slices):

        template_gaintable = gaintable.isel(solution_time=[idx])

        def refant_baselines(data: xr.DataArray) -> xr.DataArray:
            """Baseline of each antenna with refant, averaged in time."""
            return (
                data.isel(time=slc)
                .isel(
                    baseline_id=xr.DataArray(
                        baseline_index, dims="antenna_name"
                    )
                )
                .drop_vars(data.baseline_id.coords, errors="ignore")
                .mean(dim="time")
            )

        vis_refant = refant_baselines(correlated_data)
        # V(refant, antenna) -> V(antenna, refant)
        vis_refant = xr.where(
            per_antenna(conjugate), np.conj(vis_refant), vis_refant
        )
        vis_refant = xr.where(per_antenna(missing), 0.0, vis_refant)
        vis_refant = xr.where(per_antenna(is_refant), 1.0 + 0.0j, vis_refant)

        weights = refant_baselines(weight)
        weights = xr.where(per_antenna(missing), 0.0, weights)
        weights = xr.where(per_antenna(is_refant), 1.0, weights)

        baselines_table = template_gaintable.assign(
            CALPARAM_GAIN=_as_template(
                vis_refant,
                template_gaintable["CALPARAM_GAIN"],
                correlated_data.dtype,
            ),
            CALPARAM_WEIGHT=_as_template(
                weights, template_gaintable["CALPARAM_WEIGHT"], weight.dtype
            ),
        )
        baselines_table = baselines_table.rename(solution_time="time")

        delay_table_across_solutions.append(
            calculate_delays_from_gain(baselines_table, oversample)
        )

    combined_delay_table = xr.concat(delay_table_across_solutions, dim="time")

    return combined_delay_table


def _as_template(
    data: xr.DataArray, template: xr.DataArray, dtype: np.dtype
) -> xr.DataArray:
    """
    Per-antenna data ``(antenna_name, frequency, polarization)`` as a
    gaintable variable like ``template``, with its single solution time and
    coordinates. ``dtype`` is the dtype of the visibility variable the data
    comes from, which is kept as in the pre-migration implementation.
    """
    return xr.DataArray(
        data.transpose("antenna_name", "frequency", "polarization")
        .astype(dtype)
        .data[np.newaxis, ...],
        dims=template.dims,
        coords=template.coords,
    )
