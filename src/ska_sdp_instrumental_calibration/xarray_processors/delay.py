import logging

import dask
import dask.array as da
import numpy as np
import xarray as xr

from .solver import run_solver

logger = logging.getLogger()


def calculate_delay(gaintable: xr.Dataset, oversample) -> xr.Dataset:
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

    nstations = len(gaintable.antenna)

    Xgain = gaintable.gain[0, :, :, 0, 0]
    Ygain = gaintable.gain[0, :, :, 1, 1]

    Xdelay_coarse = coarse_delay(Xgain, oversample)
    Xdelay, Xoffset = update_delay(
        gaintable,
        da.zeros(nstations, dtype=float),
        Xdelay_coarse,
        pol=0,
    )

    Ydelay_coarse = coarse_delay(Ygain, oversample)
    Ydelay, Yoffset = update_delay(
        gaintable,
        da.zeros(nstations, dtype=float),
        Ydelay_coarse,
        pol=1,
    )

    return xr.Dataset(
        data_vars=dict(
            delay=(
                ["time", "antenna", "pol"],
                np.stack([Xdelay, Ydelay], axis=1).reshape(1, nstations, 2),
            ),
            offset=(
                ["time", "antenna", "pol"],
                np.stack([Xoffset, Yoffset], axis=1).reshape(1, nstations, 2),
            ),
        ),
        coords=dict(
            antenna=gaintable.antenna, pol=["XX", "YY"], time=gaintable.time
        ),
        attrs=dict(configuration=gaintable.configuration),
    )


def apply_delay_to_gaintable(
    gaintable: xr.Dataset, delay: xr.Dataset, inverse: bool = False
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
    new_gain_data = gaintable.gain.data.copy()

    Xgain = gaintable.gain[0, :, :, 0, 0]
    Ygain = gaintable.gain[0, :, :, 1, 1]

    Xdelay = delay.delay.data[0, :, 0]
    Xoffset = delay.offset.data[0, :, 0]

    Ydelay = delay.delay.data[0, :, 1]
    Yoffset = delay.offset.data[0, :, 1]

    new_gain_data[0, :, :, 0, 0] = calculate_gain_rot(
        Xgain,
        Xdelay,
        Xoffset,
        gaintable.frequency.data.reshape(1, -1),
        inverse,
    )
    new_gain_data[0, :, :, 1, 1] = calculate_gain_rot(
        Ygain,
        Ydelay,
        Yoffset,
        gaintable.frequency.data.reshape(1, -1),
        inverse,
    )

    new_gain = gaintable.gain.copy()
    new_gain.data = new_gain_data

    return gaintable.assign(
        {
            "gain": new_gain,
        }
    ).chunk(gaintable.chunks)


def update_delay(gaintable, _offset, delay, pol):
    """
    Updates the delay to the gains

    Parameters
    ----------
    gaintable: xr.Dataset
        Gaintable
    _offset: np.array
        Calculated offset per station
    delay: np.array
        Calculated delays per station
    pol: int
        Polarisations of the gains

    Returns
    -------
    np.array
        Updated delay and offset.
    """
    freq = gaintable.frequency.data.reshape(1, -1)
    gains = gaintable.gain.data[0, :, :, pol, pol]
    wgt = gaintable.weight.data[0, :, :, pol, pol]

    gains_rot = calculate_gain_rot(gains, delay, _offset, freq, inverse=True)

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


def coarse_delay(gains, oversample):
    """
    Calculates the coarse delay

    Parameters
    ----------
    frequency: xarray
        Frequency of the gains
    gains: xarray
        Gains from previous calibration step
    oversample: int
        Oversample rate

    Returns
    -------
    np.array
        Array of coarse delays for all stations
    """
    nstations = len(gains.antenna)
    nchan = len(gains.frequency)
    frequency = gains.frequency

    N = oversample * nchan
    padded_gains = da.zeros((nstations, N), "complex")

    gain_start_index = N // 2 - nchan // 2
    gain_stop_index = gain_start_index + nchan

    padded_gains[:, gain_start_index:gain_stop_index] = gains

    delay_spec = da.fft.fftshift(da.fft.fft(padded_gains, axis=1), axes=(1,))

    delay = ((1 / oversample) / (frequency[-1] - frequency[0]).data) * (
        -N // 2 + da.arange(N)
    )

    return delay[da.abs(delay_spec).argmax(axis=1)]


def calculate_gain_rot(gain, delay, offset, freq, inverse=False):
    """
    Calculates gain rotation

    Parameters
    ----------
    gain: xarray
        Gains
    delay: np.array
        Delays
    offset: np.array
        Offset
    freq: xarray
        Frequency

    Returns
    -------
    np.array
        Array of calculated gain rotation
    """

    sign = -1 if inverse else 1

    return gain * np.exp(sign * 2j * np.pi * (offset + (delay.T * freq.T))).T


def calibrate_polarization(pol, vis, modelvis, initialtable, solver):
    """
    Extract and calibrate for a single polarization

    Parameters:
    -----------
    pol: str
        Single polarization to solve for
    vis: xr.DataArray
        Visibilities
    modelvis: xr.DataArray
        Model visibilities
    initialtable: xr.Dataset
        Gaintable
    solver: func
        solver function

    Returns:
    --------
    Gaintable
    """
    scalar_vis = vis.sel(polarisation=[pol])
    scalar_model_vis = modelvis.sel(polarisation=[pol])
    scalar_table = initialtable.sel(receptor1=[pol[0]], receptor2=[pol[0]])

    logger.debug(f"Calibrating polarization {pol}")
    return run_solver(
        vis=scalar_vis,
        modelvis=scalar_model_vis,
        gaintable=scalar_table,
        solver=solver,
    )


def unstack_jones_coordinate(
    ref_gaintable: xr.Dataset, gaintable: xr.Dataset
) -> xr.Dataset:
    """Unstack Jones solutions back to diagonal of Jones matrix.

    Places stacked solutions onto the diagonal of the Jones matrix,
    preserving the reference gaintable structure.

    Parameters:
    -----------
    ref_gaintable: xr.Dataset
        intial gaintable
    gaintable: xr.Dataset
        gaintable with jones solution

    Return:
    -------
    Gaintable
    """
    new_gain_data = ref_gaintable.gain.data.copy()
    stacked_data = gaintable.gain.data

    # Place solutions on diagonal elements
    new_gain_data[..., 0, 0] = stacked_data[..., 0]
    new_gain_data[..., 1, 1] = stacked_data[..., 1]

    new_gain = ref_gaintable.gain.copy(data=new_gain_data)

    return ref_gaintable.assign({"gain": new_gain}).chunk(ref_gaintable.chunks)


def stack_jones_coordinate(gaintable: xr.Dataset) -> xr.Dataset:
    """Stack receptor1 and receptor2 into Jones_Solutions coordinates.

    Transforms individual polarization components (XX, YY, etc.) into
    a single Jones_Solutions dimension with polarization labels.

    Parameters:
    -----------
    gaintable: xr.Dataset
        Gaintable with all polarizations

    Return:
    -------
    Gaintable
    """
    stacked = gaintable.stack(Jones_Solutions=("receptor1", "receptor2"))

    # Extract polarization strings from stacked receptor pairs
    receptors = stacked["Jones_Solutions"].values
    polstrs = [f"J_{p1}{p2}".upper() for p1, p2 in receptors]

    return stacked.drop_vars(
        ["Jones_Solutions", "receptor1", "receptor2"]
    ).assign_coords({"Jones_Solutions": polstrs})
