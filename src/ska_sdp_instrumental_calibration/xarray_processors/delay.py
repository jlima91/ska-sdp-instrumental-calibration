import logging
import warnings
from functools import partial

import numpy as np
import xarray as xr
from numpy.exceptions import ComplexWarning
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.visibility import Visibility

from ._utils import with_chunks

logger = logging.getLogger()


class DelayTable(xr.Dataset):
    """
    Container for per-antenna, per-polarisation delay solutions computed
    from a GainTable.

    **Coordinates**

    - time: time centroids of solutions, in seconds elapsed since the MJD
      reference epoch, ``[ntimes]``.

    - antenna: integer antenna indices starting at 0, ``[nants]``.

    - pol: polarisation labels, e.g. ``['XX', 'YY']``, ``[npol]``.

    **Data variables**

    - delay: per-antenna, per-polarisation group-delay in seconds,
      real-valued ``[ntimes, nants, npol]``.

    - offset: per-antenna, per-polarisation phase offset in cycles,
      real-valued ``[ntimes, nants, npol]``.

    **Attributes**

    - data_model: name of this class, used internally for saving to / loading
      from files.

    - configuration: Array configuration as a Configuration object.

    Here is an example::

        <xarray.DelayTable>
        Dimensions:  (time: 1, antenna: 115, pol: 2)
        Coordinates:
          * time     (time) float64 5.085e+09
          * antenna  (antenna) int64 0 1 2 ... 113 114
          * pol      (pol) <U2 'XX' 'YY'
        Data variables:
            delay    (time, antenna, pol) float64 ...
            offset   (time, antenna, pol) float64 ...
        Attributes:
            data_model:     DelayTable
            configuration:  <xarray.Configuration>
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
        antenna: np.ndarray,
        pol: list[str],
        configuration: Configuration | None = None,
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
        antenna
            Integer antenna indices ``[nants]``.
        pol
            Polarisation labels, e.g. ``['XX', 'YY']``.
        configuration
            Configuration object describing the array configuration.
        """
        coords = {
            "time": time,
            "antenna": antenna,
            "pol": list(pol),
        }
        datavars = {
            "delay": xr.DataArray(delay, dims=["time", "antenna", "pol"]),
            "offset": xr.DataArray(offset, dims=["time", "antenna", "pol"]),
        }
        attrs = {
            "data_model": "DelayTable",
            "configuration": configuration,
        }
        return cls(data_vars=datavars, coords=coords, attrs=attrs)


def calculate_delays_from_gain(
    gaintable: GainTable, oversample: int
) -> DelayTable:
    """
    Applies the delay to the given gaintable

    Parameters
    ----------
    gaintable
        Gaintable
    oversample
        Oversample rate required for the delay

    Returns
    -------
        A dataset holding delay data
    """
    # Delay fit needs the whole band, so only frequency is rechunked.
    # Time and antenna keep the gaintable chunks, so that delays for
    # all antennas in a chunk are calculated in a single task.
    gain = gaintable["gain"].chunk(frequency=-1)
    weight = gaintable["weight"].chunk(frequency=-1)

    # We only calculate delays based on diagonal terms
    pols = [
        f"{gaintable['receptor1'][idx].item()}"
        f"{gaintable['receptor2'][idx].item()}"
        for idx in (0, 1)
    ]

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
                ["frequency", "receptor1", "receptor2"],
                ["frequency", "receptor1", "receptor2"],
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
        delay=delay.transpose("time", "antenna", "pol").data,
        offset=offset.transpose("time", "antenna", "pol").data,
        time=gaintable["time"].values,
        antenna=gaintable["antenna"].values,
        pol=pols,
        configuration=gaintable.attrs["configuration"],
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

    Delays are calculated at once for all leading dimensions
    (time, antenna) of the chunk, using only the diagonal
    terms of the 2x2 gain matrix.

    Parameters
    ----------
    gain: (time, nant, freq, nrec, nrec) (np.complex64)
    weight: (time, nant, freq, nrec, nrec) (float)
    frequency: (freq,) (float)
    oversample: int

    Returns
    -------
        delay: (time, nant, pol)
        offset: (time, nant, pol)
    """
    # Pick diagonal terms, keeping frequency as the last axis
    # Shape: (time, nant, pol, freq)
    gain = np.stack([gain[..., 0, 0], gain[..., 1, 1]], axis=-2)
    weight = np.stack([weight[..., 0, 0], weight[..., 1, 1]], axis=-2)

    delay_coarse = coarse_delay(gain, frequency, oversample)
    delay, offset = update_delay(
        gain,
        weight,
        frequency,
        np.zeros_like(delay_coarse),
        delay_coarse,
    )

    # Remove the trailing size 1 axis left by the numpy functions
    return delay[..., 0], offset[..., 0]


def apply_delay_to_gaintable(
    gaintable: GainTable, delaytable: DelayTable, inverse: bool = False
) -> GainTable:
    """
    Applies the delay to the given gaintable

    Parameters
    ----------
    gaintable
        Gaintable on which we need to apply delays
    delaytable
        DelayTable which holds calculated delays
    inverse
        Whether to invert the delayes before appluing

    Returns
    -------
        Gaintable with updated gains
    """
    # gaintable stores gains in 2x2 matrix (receptor1, receptor2)
    # while delaytable stores them in 1x2 array (pol).
    # Both are core dimensions, so a single task rotates all
    # polarisations of a gaintable chunk.
    # Gains are merged across frequency, so that the whole band is
    # processed in a single task, instead of a task per frequency chunk.
    # Frequency is passed as an argument (not in kwargs), so that
    # dask aligns it with the frequency chunks of the gains.
    delay_rotated_gain = xr.apply_ufunc(
        _apply_delay_ufunc_,
        gaintable["gain"].chunk(frequency=-1),
        delaytable["delay"],
        delaytable["offset"],
        gaintable["frequency"],
        input_core_dims=[["receptor1", "receptor2"], ["pol"], ["pol"], []],
        output_core_dims=[["receptor1", "receptor2"]],
        output_dtypes=[gaintable["gain"].dtype],
        dask="parallelized",
        kwargs=dict(inverse=inverse),
    )

    # Restore the chunks of the input gaintable
    return gaintable.assign(
        gain=with_chunks(
            delay_rotated_gain.transpose(*gaintable["gain"].dims),
            gaintable["gain"].chunksizes,
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
    gain: (time, nant, freq, nrec, nrec) (complex)
    delay: (time, nant, 1, pol) (float)
        apply_ufunc inserts a size 1 axis for frequency
    offset: (time, nant, 1, pol) (float)
    frequency: (freq,) (float)
    inverse: bool
        Whether to invert the delays before applying

    Returns
    -------
        Gains with delays applied, same shape and dtype as gain
    """
    logger.info(
        "apply_delay_ufunc: gain shape=%s, delay shape=%s, "
        "offset shape=%s, frequency shape=%s, inverse=%s",
        gain.shape,
        delay.shape,
        offset.shape,
        frequency.shape,
        inverse,
    )
    # Shape: (time, nant, freq, pol)
    gain_diagonal = np.stack([gain[..., 0, 0], gain[..., 1, 1]], axis=-1)

    rotated_diagonal = calculate_gain_rot(
        gain_diagonal,
        delay,
        offset,
        frequency[..., np.newaxis],
        inverse=inverse,
    )

    rotated_gain = np.zeros_like(gain)
    rotated_gain[..., 0, 0] = rotated_diagonal[..., 0]
    rotated_gain[..., 1, 1] = rotated_diagonal[..., 1]

    return rotated_gain


def update_delay(
    gains: np.ndarray,
    wgt: np.ndarray,
    freq: np.ndarray,
    _offset: np.ndarray,
    delay: np.ndarray,
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Updates the delay to the gains

    Parameters
    ----------
    gains: np.ndarray. complex. (..., freq)
    wgt: np.ndarray. float (..., freq)
    freq: np.ndarray. float (freq,)
    _offset: np.ndarray. float (..., 1)
        Calculated offset per station
    delay: np.ndarray. float (..., 1)
        Calculated delays per station

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


def create_delaytable_from_vis(
    vis: Visibility, gaintable: GainTable, refant: int, oversample: int
) -> DelayTable:
    """
    Calculates delays from visibility data by processing each solution interval

    Parameters
    ----------
    vis
        Visibility data. If backed by a dask array, can be chunked in time
        and frequency axis.
    gaintable
        Gaintable containing solution intervals.
    refant
        Reference antenna
    oversample
        Oversample rate required for the delay

    Returns
    -------
        Dataset of calculated delays
    """

    baseline_ids = vis.baselineid[
        (vis.antenna1 == refant) | (vis.antenna2 == refant)
    ]

    gaintable = gaintable.rename(time="solution_time")

    soln_interval_slices = gaintable.soln_interval_slices

    delay_table_across_solutions = []

    for idx, slc in enumerate(soln_interval_slices):

        vis_per_solution = vis.isel(time=slc)

        template_gaintable = gaintable.isel(solution_time=[idx])

        vis_refant = (
            vis_per_solution["vis"]
            .isel(baselineid=baseline_ids)
            .mean(dim="time")
        )

        vis_refant[refant:, ...] = vis_refant[refant:, ...].conj()

        weights = (
            vis_per_solution["weight"]
            .isel(baselineid=baseline_ids)
            .mean(dim="time")
        )

        vis_refant[refant, ...] = 1.0 + 0.0j
        weights[refant, ...] = 1.0

        vis_refant_data = vis_refant.data.reshape(
            template_gaintable["gain"].shape
        )
        weight_data = weights.data.reshape(template_gaintable["weight"].shape)

        reshaped_vis_refant = xr.DataArray(
            vis_refant_data,
            dims=template_gaintable["gain"].dims,
        )
        reshaped_weights = xr.DataArray(
            weight_data,
            dims=template_gaintable["weight"].dims,
        )
        baselines_table = template_gaintable.assign(
            gain=reshaped_vis_refant,
            weight=reshaped_weights,
        )
        baselines_table = baselines_table.rename(solution_time="time")

        delay_table_across_solutions.append(
            calculate_delays_from_gain(baselines_table, oversample)
        )

    combined_delay_table = xr.concat(delay_table_across_solutions, dim="time")

    return combined_delay_table
