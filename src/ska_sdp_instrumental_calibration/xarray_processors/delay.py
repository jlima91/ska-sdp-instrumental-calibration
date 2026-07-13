import logging
from typing import Optional, Sequence

import dask.array as da
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.configuration import Configuration

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
        delay: NDArray,
        offset: NDArray,
        time: NDArray,
        antenna: NDArray,
        pol: Sequence[str],
        configuration: Optional[Configuration] = None,
    ):
        """
        Create a DelayTable instance directly from numpy arrays.

        :param delay: Per-antenna, per-polarisation group-delay in seconds
            ``[ntimes, nants, npol]``
        :type delay: ndarray

        :param offset: Per-antenna, per-polarisation phase offset in cycles
            ``[ntimes, nants, npol]``
        :type offset: ndarray

        :param time: Centroids of solutions, in seconds elapsed since the MJD
            reference epoch ``[ntimes]``
        :type time: ndarray

        :param antenna: Integer antenna indices ``[nants]``
        :type antenna: ndarray

        :param pol: Polarisation labels, e.g. ``['XX', 'YY']``
        :type pol: sequence of str

        :param configuration: Configuration object describing the array
            configuration
        :type configuration: Configuration or None, optional
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
        return cls(datavars, coords=coords, attrs=attrs)

    def __sizeof__(self):
        """Override default method to return size of dataset
        :return: int
        """
        return int(self.nbytes)


def calculate_delays_from_gain(gaintable: GainTable, oversample) -> DelayTable:
    """
    Applies the delay to the given gaintable

    Parameters
    ----------
    gaintable: xr.Dataset
        Gaintable
    oversample: int
        Oversample rate required for the delay

    Returns
    -------
    xr.DataSet
        delay
    """
    nstations = gaintable["antenna"].size

    apply_ufunc_results = {"delay": {}, "offset": {}}

    # We calculate delays only for XX and YY terms
    for pol, receptor1, receptor2 in (("XX", 0, 0), ("YY", 1, 1)):
        gain = gaintable["gain"][0, :, :, receptor1, receptor2].chunk(
            antenna=1, frequency=-1
        )
        weight = gaintable["weight"][0, :, :, receptor1, receptor2].chunk(
            antenna=1, frequency=-1
        )
        initial_offset = xr.zeros_like(gaintable["antenna"]).chunk(antenna=1)

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
                frequency=gaintable["frequency"].data, oversample=oversample
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
            axis=1,
        ).reshape(1, nstations, 2),
        offset=da.stack(
            [
                apply_ufunc_results["offset"]["XX"],
                apply_ufunc_results["offset"]["YY"],
            ],
            axis=1,
        ).reshape(1, nstations, 2),
        time=gaintable["time"],
        antenna=gaintable["antenna"],
        pol=["XX", "YY"],
        configuration=gaintable.attrs["configuration"],
    )


def _calculate_delays_ufunc_(
    gain: np.ndarray,
    weight: np.ndarray,
    offset: np.ndarray,
    frequency: np.ndarray,
    oversample: int,
):
    """
    gain: np.ndarray (nant, freq) (np.complex64)
    weight: np.ndarray (nant, freq) (float)
    offset: np.ndarray (nant,) (float)
    frequency: np.ndarray (freq) (float)
    oversample: int

    In apply_ufunc call, the dataset might be chunked in antenna
    So after distribution, nant will become 1
    """
    # Remove extra antenna dimension
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

    return delay, offset


def apply_delay_to_gaintable(
    gaintable: xr.Dataset, delaytable: xr.Dataset, inverse: bool = False
) -> xr.Dataset:
    """
    Applies the delay to the given gaintable

    Parameters
    ----------
    gaintable: xr.Dataset
        Gaintable
    oversample: int
        Oversample rate required for the delay

    Returns
    -------
    xr.Dataset
        Gaintable with updated gains
    """
    new_gains = gaintable["gain"].copy()

    # We calculate delays only for XX and YY terms
    for pol, receptor1, receptor2 in (("XX", 0, 0), ("YY", 1, 1)):
        gain = gaintable["gain"][..., receptor1, receptor2]
        frequency = gaintable["frequency"]
        delay = delaytable["delay"].sel(pol=pol)
        offset = delaytable["offset"].sel(pol=pol)

        delay_rotated_gain = xr.apply_ufunc(
            calculate_gain_rot,
            gain,
            delay,
            offset,
            frequency,
            output_dtypes=[np.complex64],
            dask="parallelized",
            kwargs=dict(inverse=inverse),
        )

        new_gains[..., receptor1, receptor2] = delay_rotated_gain

    return gaintable.assign(gain=new_gains)


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
    _offset: np.ndarray. float (,)
        Calculated offset per station
    delay: np.ndarray. float (,)
        Calculated delays per station

    Returns
    -------
        Updated delay and offset
        np.ndarray, with last dimension of size 1
    """
    gains_rot = calculate_gain_rot(gains, delay, _offset, freq, inverse=True)

    # cycles = calculate_cycles(gains_rot)
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


# def calculate_cycles(gains_rot):
#     return np.unwrap(np.angle(gains_rot)) / (2 * np.pi)


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
        np.ndarray, with last dimension of size 1
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

    return delay[..., np.abs(delay_spec).argmax(axis=-1, keepdims=True)]


def calculate_gain_rot(
    gain: np.ndarray,
    delay: float,
    offset: float,
    freq: np.ndarray,
    inverse=False,
):
    """
    Calculates gain rotation.
    The function assumes that for numpy arrays as input, the values
    are broadcastable across dimensions excluding the last one.

    Parameters
    ----------
    gain: np.ndarray. (complex64)
    delay: float
    offset: float
    freq: np.ndarray. (float)
    inverse: bool

    Returns
    -------
    np.ndarray (complex64)
        Array of calculated gain rotation
        Same shape and dtype as gain
    """

    sign = -1 if inverse else 1
    return gain * np.exp(sign * 2j * np.pi * (offset + (delay * freq)))


def calculate_delays_from_vis(vis: xr.Dataset, refant: int) -> DelayTable:
    """
    Calculates delays from visibility data

    Parameters
    ----------
    vis
        Visibility data
    refant
        Reference antenna

    Returns
    -------
        Dataset of calculated delays
    """
    raise NotImplementedError(
        "Calculating delays from visibility data is not implemented yet."
    )
