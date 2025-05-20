"""Post-calibration fits."""

__all__ = ["model_rotations"]

from pathlib import Path

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from astropy import constants as const

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.post_processing")


def _plot_rm(
    rm_peek: np.float64,
    rm_spec: npt.NDArray[np.complex128],
    rm_vals: npt.NDArray[np.float64],
    title: str,
    plot_path: Path | str,
):
    """
    Plot the rotation measure spectrum for a station.

    Parameters
    ----------
    rm_peek : float
        The peak value of the rotation measure.
    rm_spec : array-like of complex
        The complex valued rotation measure spectrum.
    rm_vals : array-like of float
        The possible values of rotation measure.
    title : str
        The title of the plot.
    plot_path : Path or str
        The path to save the plot to.
    """
    xlim = 3 * np.max(np.abs(rm_peek))

    plt.figure()

    ax = plt.subplot(111)
    ax.set_title(title)
    ax.set_xlabel("RM (rad / m^2)")
    ax.plot(rm_vals, np.abs(rm_spec), "b", label="abs")
    ax.plot(rm_vals, np.real(rm_spec), "c", label="re")
    ax.plot(rm_vals, np.imag(rm_spec), "m", label="im")
    ax.plot(rm_peek * np.ones(2), ax.get_ylim(), "b--")
    ax.set_xlim((-xlim, xlim))
    ax.grid()
    ax.legend()

    plt.savefig(plot_path)


@dask.delayed
def _get_rm_peeks(
    rm_spec: npt.NDArray[np.complex128],
    rm_vals: npt.NDArray[np.float64],
    peak_threshold: float,
    nstations: int,
    plot_path_prefix: str,
    plot_sample: bool = False,
):
    """
    Get the rotation measure values for each station in the gain table.

    Parameters
    ----------
    rm_spec : array-like of complex
        The rotation measure spectrum for each station.
    rm_vals : array-like of float
        The possible values of rotation measure.
    peak_threshold : float, optional
        The minimum absolute amplitude of the rotation measure spectrum
        required to consider it a valid peak.
    nstations : int, optional
        The number of stations in the gain table.
    plot_path_prefix : str, optional
        The prefix to be used for the plots.
    plot_sample : bool, optional
        Whether to plot the rotation measure spectrum for a sample station.

    Returns
    -------
    rm_peeks : array-like of float
        The rotation measure values for each station in the gain table.
    """
    stn = nstations - 1
    rm_peeks = np.zeros(nstations)

    update_indexes = np.where(np.max(np.abs(rm_spec), axis=1) > peak_threshold)
    rm_peeks[update_indexes] = rm_vals[
        np.argmax(np.abs(rm_spec[update_indexes]), axis=1)
    ]

    if plot_sample:
        _plot_rm(
            rm_peeks[stn],
            rm_spec[stn],
            rm_vals,
            f"RM spectrum for station {stn}",
            f"{plot_path_prefix}_{stn}.png",
        )

    return rm_peeks


def model_rotations(
    gaintable: xr.Dataset,
    peak_threshold: float = 0.5,
    plot_sample: bool = False,
    plot_path_prefix: str = "./",
    ref: int = 0,
    oversample: int = 5,
) -> xr.Dataset:
    """
    Fit a rotation measure for each station Jones matrix.

    For each station, the Jones matrix for each channel is used to
    operate on a unit vector. The result is expressed as a complex
    number, and the spectrum of complex numbers is the Fourier
    transformed with respect to wavelength squared. The peaks of this
    transformed spectrum is taken as the rotation measure for the
    station, and used to initialise a new gaintable.

    Parameters
    ----------
    gaintable : xr.Dataset
        GainTable dataset to be to modelled.
    peak_threshold : float
        Height of peak in the RM spectrum required for a
        rotation detection.
    plot_sample : bool
        Whether or not to plot a sample RM spectrum.
    plot_path_prefix : str
        Path prefix to save the plots.
    ref : int
        Reference station to rotate against.
    oversample : int
        Oversampling factor for the rotation measure spectrum.

    Returns
    -------
    new_gaintable : xr.Dataset
        New GainTable dataset with model rotations.
    """
    nstations = len(gaintable.antenna)
    nfreq = len(gaintable.frequency)

    lambda_sq = (
        const.c.value / gaintable.frequency  # pylint: disable=no-member
    ) ** 2

    rm_res = 1 / oversample / (np.max(lambda_sq) - np.min(lambda_sq))
    rm_max = 1 / (lambda_sq[-2] - lambda_sq[-1])
    rm_vals = da.arange(-rm_max, rm_max, rm_res)

    uvec = np.array([1, 0])

    invN = 1 / nfreq
    invR = 1 / np.sqrt(2)

    reljones = da.einsum(
        "sfpx,fqx->sfpq",
        gaintable.gain[0, :],
        gaintable.gain[0, ref].conj(),
    )
    reljones /= invR * da.linalg.norm(reljones, axis=(2, 3), keepdims=True)

    rvec = da.einsum("sfpq,q->sfp", reljones, uvec)
    f_spec = rvec[:, :, 0] + 1j * rvec[:, :, 1]

    rm_spec = invN * da.einsum(
        "rf,sf->sr",
        np.exp(np.einsum("i,j->ij", -1j * rm_vals, lambda_sq)),
        f_spec,
    )

    rm_peeks = da.from_delayed(
        _get_rm_peeks(
            rm_spec,
            rm_vals,
            peak_threshold,
            nstations,
            plot_path_prefix,
            plot_sample,
        ),
        (nstations,),
        rm_vals.dtype,
    )

    peek_diff = rm_peeks - rm_peeks[ref]
    d_pa = lambda_sq.data * peek_diff.reshape(nstations, -1)

    updated_gain = da.stack(
        (np.cos(d_pa), -np.sin(d_pa), np.sin(d_pa), np.cos(d_pa)),
        axis=2,
    ).reshape(1, nstations, nfreq, 2, 2).astype(gaintable.gain.dtype)

    updated_gain_xdr = gaintable.gain.copy()
    updated_gain_xdr.data = updated_gain

    return gaintable.assign({"gain": updated_gain_xdr}).chunk(gaintable.chunks)
