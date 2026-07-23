import logging
import warnings

import dask.array as da
import numpy as np
import xarray as xr
from numpy.exceptions import ComplexWarning
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.configuration import Configuration

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
    ntime = gaintable.sizes["time"]
    nant = gaintable.sizes["antenna"]
    # We only calculate delays based on diagonal terms
    pols = ["XX", "YY"]

    gain_gain_chunked = gaintable["gain"].chunk(
        time=1, antenna=1, frequency=-1
    )
    gain_weight_chunk = gaintable["weight"].chunk(
        time=1, antenna=1, frequency=-1
    )

    apply_ufunc_results = {"delay": {}, "offset": {}}
    # We calculate delays only for XX and YY terms
    for pol, rec1idx, rec2idx in (("XX", 0, 0), ("YY", 1, 1)):
        gain = gain_gain_chunked[..., rec1idx, rec2idx]
        weight = gain_weight_chunk[..., rec1idx, rec2idx]
        initial_offset = xr.zeros_like(gaintable["antenna"]).chunk(antenna=1)

        with warnings.catch_warnings():
            # apply_ufunc throws a false warning, when it detects that
            # one of the inputs (gain) has complex dtype, but outputs
            # are all non-complex values
            warnings.simplefilter("ignore", ComplexWarning)

            delay, offset = xr.apply_ufunc(
                _calculate_delays_ufunc_,
                gain,
                weight,
                initial_offset,
                input_core_dims=[["frequency"], ["frequency"], []],
                output_core_dims=[[], []],
                output_dtypes=[np.float64, np.float64],
                dask="parallelized",
                kwargs=dict(
                    frequency=gaintable["frequency"].data,
                    oversample=oversample,
                ),
            )

        apply_ufunc_results["delay"][pol] = delay
        apply_ufunc_results["offset"][pol] = offset

    return DelayTable.constructor(
        delay=da.stack(
            [
                apply_ufunc_results["delay"]["XX"],
                apply_ufunc_results["delay"]["YY"],
            ],
            axis=-1,
        ).reshape(ntime, nant, 2),
        offset=da.stack(
            [
                apply_ufunc_results["offset"]["XX"],
                apply_ufunc_results["offset"]["YY"],
            ],
            axis=-1,
        ).reshape(ntime, nant, 2),
        time=gaintable["time"].values,
        antenna=gaintable["antenna"].values,
        pol=pols,
        configuration=gaintable.attrs["configuration"],
    )


def _calculate_delays_ufunc_(
    gain: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
    frequency: np.ndarray,
    oversample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function acts as a bridge between
    calculate_delays_from_gain and numpy based
    delay functions (coarse_delay, update_delay)

    In apply_ufunc call, the dataset is chunked in time
    and antenna, with chunksize=1,
    So in this function, both time and nant will be 1

    Parameters
    ----------
    gain: (time, nant, freq,) (np.complex64)
    weight: (time, nant, freq,) (float)
    offset: (time, nant,) (float)
    frequency: (freq,) (float)
    oversample: int

    Returns
    -------
        delay: (time, nant,)
        offset: (time, nant,)
    """
    # Remove extra time and antenna dimension
    gain = gain.flatten()
    weight = weight.flatten()
    offset = offset.flatten()

    delay_coarse = coarse_delay(gain, frequency, oversample)
    delay, offset = update_delay(
        gain,
        weight,
        frequency,
        offset,
        delay_coarse,
    )

    # Bring back original (time, antenna) shape
    # for apply_ufunc to broadcast nicely
    # Note that delay and offset are already numpy arrays
    # with shape (1,). So we need to add only one new axis.
    return delay[np.newaxis], offset[np.newaxis]


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
    new_gains = gaintable["gain"].copy()

    # We calculate delays only for XX and YY terms
    for pol, rec1idx, rec2idx in (("XX", 0, 0), ("YY", 1, 1)):
        gain = gaintable["gain"][..., rec1idx, rec2idx]
        frequency = gaintable["frequency"]
        delay = delaytable["delay"].sel(pol=pol)
        offset = delaytable["offset"].sel(pol=pol)

        # calculate_gain_rot can operate on a single gain value
        # or a chunk along frequency dimension
        # so no need to reshape/broadcast (no input_core_dims)
        # Internally, this relies on dask.array.blockwise "align_arrays"
        # which will align chunks across broadcasted dimensions
        delay_rotated_gain = xr.apply_ufunc(
            calculate_gain_rot,
            gain,
            delay,
            offset,
            frequency,
            output_dtypes=[gain.dtype],
            dask="parallelized",
            kwargs=dict(inverse=inverse),
        )

        new_gains[..., rec1idx, rec2idx] = delay_rotated_gain

    return gaintable.assign(gain=with_chunks(new_gains, gaintable.chunks))


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
    gains: np.ndarray. complex. (freq,)
    wgt: np.ndarray. float (freq,)
    freq: np.ndarray. float (freq,)
    _offset: np.ndarray. float (1,)
        Calculated offset per station
    delay: np.ndarray. float (1,)
        Calculated delays per station

    Returns
    -------
        Updated delay and offset
        Each a np.ndarray, Shape: (1,)
    """
    gains_rot = calculate_gain_rot(gains, delay, _offset, freq, inverse=True)

    cycles = np.unwrap(np.angle(gains_rot)) / (2 * np.pi)

    denom = np.sum(wgt) * np.sum(wgt * freq * freq) - np.sum(
        wgt * freq
    ) * np.sum(wgt * freq)

    offset = (
        _offset
        + (
            np.sum(wgt * freq * freq) * np.sum(wgt * cycles)
            - np.sum(wgt * freq) * np.sum(wgt * freq * cycles)
        )
        / denom
    )

    updated_delay = (
        np.sum(wgt) * np.sum(wgt * cycles * freq)
        - np.sum(wgt * cycles) * np.sum(wgt * freq)
    ) / denom

    return delay + updated_delay, offset


def coarse_delay(
    gains: np.ndarray, frequency: np.ndarray, oversample: int
) -> np.ndarray[float]:
    """
    Calculates the coarse delay

    Parameters
    ----------
    gains: np.ndarray. complex. (freq,)
    frequency: np.ndarray. float (freq,)
    oversample: int
        Oversample rate

    Returns
    -------
        Delay value for given frequency range
        np.ndarray, Shape (1,)
    """
    nchan = frequency.size
    N = oversample * nchan
    gain_start_index = N // 2 - nchan // 2
    gain_stop_index = gain_start_index + nchan

    padded_gains = np.zeros((N,), gains.dtype)
    padded_gains[gain_start_index:gain_stop_index] = gains

    delay_spec = np.fft.fftshift(np.fft.fft(padded_gains))

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
    vis: xr.Dataset, gaintable: xr.Dataset, refant: int, oversample: int
) -> xr.Dataset:
    """ "
    Calculates delays from visibility data by processing each solution interval

    Parameters
    ----------
    vis: xarray
        Visibility data. If backed by a dask array, can be chunked in time
        and frequency axis.
    gaintable: xarray
        Gaintable containing solution intervals.
    refant: int
        Reference antenna
    oversample: int
        Oversample rate required for the delay

    Returns
    -------
    xr.Dataset
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
