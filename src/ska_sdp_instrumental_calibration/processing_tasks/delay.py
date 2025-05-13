import dask
import dask.array as da
import numpy as np
import xarray as xr


def apply_delay(gaintable: xr.Dataset, oversample) -> xr.Dataset:
    """
    Applies the delay to the given gaintable

    Parameters:
    -----------
        gaintable: xr.Dataset
            Gaintable
        oversample: int
            Oversample rate required for the delay
    Returns:
    --------
        gaintable: xr.Dataset
            Gaintable with updated gains
    """
    nstations = len(gaintable.antenna)
    new_gain_data = gaintable.gain.data.copy()

    Xgain = new_gain_data[0, :, :, 0, 0]
    Ygain = new_gain_data[0, :, :, 1, 1]

    Xdelay_coarse = coarse_delay(gaintable.frequency, Xgain, oversample)
    Xdelay, Xoffset = update_delay(
        gaintable,
        da.zeros(nstations, dtype=float),
        Xdelay_coarse,
        pol=0,
    )

    Ydelay_coarse = coarse_delay(gaintable.frequency, Ygain, oversample)
    Ydelay, Yoffset = update_delay(
        gaintable,
        da.zeros(nstations, dtype=float),
        Ydelay_coarse,
        pol=1,
    )

    new_gain_data[0, :, :, 0, 0] = calculate_gain_rot(
        Xgain, Xdelay, Xoffset, gaintable.frequency.data.reshape(1, -1)
    )
    new_gain_data[0, :, :, 1, 1] = calculate_gain_rot(
        Ygain, Ydelay, Yoffset, gaintable.frequency.data.reshape(1, -1)
    )

    new_gain = gaintable.gain.copy()
    new_gain.data = new_gain_data

    return gaintable.assign({"gain": new_gain}).chunk(gaintable.chunks)


def update_delay(gaintable, _offset, delay, pol):
    """
    Updates the delay to the gains

    Parameters:
    -----------
        gaintable: xr.Dataset
            Gaintable
        _offset: np.array
            Calculated offset per station
        delay: np.array
            Calculated delays per station
        pol: int
            Polarisations of the gains
    Returns:
    --------
        np.array
            Updated delay and offset.
    """
    freq = gaintable.frequency.data.reshape(1, -1)
    gains = gaintable.gain.data[0, :, :, pol, pol]
    wgt = gaintable.weight.data[0, :, :, pol, pol]

    gains_rot = calculate_gain_rot(gains, delay, _offset, freq)

    cycles = da.from_delayed(
        calculate_cycles(gains_rot), gains.shape, gains_rot.real.dtype
    )

    denom = np.sum(wgt, axis=1) * np.sum(wgt * freq * freq, axis=1) - np.sum(
        wgt * freq, axis=1
    ) * np.sum(wgt * freq, axis=1)
    offset = (
        _offset
        + (
            np.sum(wgt * freq * freq, axis=1) * np.sum(wgt * cycles, axis=1)
            - np.sum(wgt * freq, axis=1) * np.sum(wgt * freq * cycles, axis=1)
        )
        / denom
    )

    updated_delay = (
        np.sum(wgt, axis=1) * np.sum(wgt * cycles * freq, axis=1)
        - np.sum(wgt * cycles, axis=1) * np.sum(wgt * freq, axis=1)
    ) / denom
    return delay + updated_delay, offset


@dask.delayed
def calculate_cycles(gains_rot):
    return np.unwrap(np.angle(gains_rot)) / (2 * np.pi)


def coarse_delay(frequency, gains, oversample):
    """
    Calculates the coarse delay

    Parameters:
    -----------
        frequency: xarray
            Frequency of the gains
        gains: xarray
            Gains from previous calibration step
        oversample: int
            Oversample rate
    Returns:
    ---------
        np.array
            Array of coarse delays for all stations

    """
    nstations, nchan = gains.shape
    N = oversample * nchan
    padded_gains = da.zeros((nstations, N), "complex")
    gain_start_index = N // 2 - nchan // 2
    gain_stop_index = N // 2 + nchan // 2
    padded_gains[:, gain_start_index:gain_stop_index] = gains
    delay_spec = da.fft.fftshift(da.fft.fft(padded_gains, axis=1), axes=(1,))

    delay = ((1 / oversample) / (frequency[-1] - frequency[0]).data) * (
        -N // 2 + da.arange(N)
    )

    return delay[da.abs(delay_spec).argmax(axis=1)]


def calculate_gain_rot(gain, delay, offset, freq):
    """
    Calculates gain rotation

    Parameters:
    -----------
        gain: xarray
            Gains
        delay: np.array
            Delays
        offset: np.array
            Offset
        freq: xarray
            Frequency
    Returns:
    ---------
        np.array
            Array of calculated gain rotation
    """
    return gain * np.exp(-2j * np.pi * (offset + (delay.T * freq.T))).T
