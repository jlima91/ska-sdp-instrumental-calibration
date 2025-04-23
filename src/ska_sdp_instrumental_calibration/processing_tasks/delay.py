import numpy as np
import xarray as xr


def apply_delay(gaintable: xr.Dataset, oversample) -> xr.Dataset:
    Xgain = gaintable.gain.data[0, :, :, 0, 0]
    Ygain = gaintable.gain.data[0, :, :, 1, 1]

    Xdelay_coarse = coarse_delay(gaintable.frequency, Xgain, oversample)
    Xdelay, Xoffset = update_delay(
        gaintable, np.zeros(20, dtype=float), Xdelay_coarse, pol=0
    )

    Ydelay_coarse = coarse_delay(gaintable.frequency, Ygain, oversample)
    Ydelay, Yoffset = update_delay(
        gaintable, np.zeros(20, dtype=float), Ydelay_coarse, pol=1
    )
    Xgain = calculate_gain_rot(
        Xgain, Xdelay, Xoffset, gaintable.frequency.data.reshape(1, -1)
    )
    Ygain = calculate_gain_rot(
        Ygain, Ydelay, Yoffset, gaintable.frequency.data.reshape(1, -1)
    )

    gaintable["gain"].data[0, :, :, 0, 0] = Xgain
    gaintable["gain"].data[0, :, :, 1, 1] = Ygain

    return gaintable


def update_delay(gaintable, offset, delay, pol):
    freq = gaintable.frequency.data.reshape(1, -1)
    gains = gaintable.gain.data[0, :, :, pol, pol]
    wgt = gaintable.weight.data[0, :, :, pol, pol]

    gains_rot = calculate_gain_rot(gains, delay, offset, freq)

    cycles = np.unwrap(np.angle(gains_rot)) / (2 * np.pi)

    denom = np.sum(wgt, axis=1) * np.sum(wgt * freq * freq, axis=1) - np.sum(
        wgt * freq, axis=1
    ) * np.sum(wgt * freq, axis=1)
    offset += (
        np.sum(wgt * freq * freq, axis=1) * np.sum(wgt * cycles, axis=1)
        - np.sum(wgt * freq, axis=1) * np.sum(wgt * freq * cycles, axis=1)
    ) / denom

    updated_delay = (
        np.sum(wgt, axis=1) * np.sum(wgt * cycles * freq, axis=1)
        - np.sum(wgt * cycles, axis=1) * np.sum(wgt * freq, axis=1)
    ) / denom
    return delay + updated_delay, offset


def coarse_delay(frequency, gains, oversample):
    nstations, nchan = gains.shape
    N = oversample * nchan
    padded_gains = np.zeros((nstations, N), "complex")
    gain_start_index = N // 2 - nchan // 2
    gain_stop_index = N // 2 + nchan // 2
    padded_gains[:, gain_start_index:gain_stop_index] = gains
    delay_spec = np.fft.fftshift(np.fft.fft(padded_gains, axis=1), axes=(1,))

    delay = ((1 / oversample) / (frequency[-1] - frequency[0]).data) * (
        -N // 2 + np.arange(N)
    )

    return delay[np.abs(delay_spec).argmax(axis=1)]


def calculate_gain_rot(gain, delay, offset, freq):
    return gain * np.exp(-2j * np.pi * (offset + (delay.T * freq.T))).T
