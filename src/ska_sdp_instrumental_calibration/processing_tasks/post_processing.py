"""Post-calibration fits."""

__all__ = ["model_rotations"]

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy import constants as const

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.post_processing")


def model_rotations(
    gaintable: xr.Dataset,
    peak_threshold: float = 0.5,
    plot_sample: bool = False,
    plot_path_prefix: str = "./",
) -> xr.Dataset:
    """Fit a rotation measure for each station Jones matrix.

    For each station, the Jones matrix for each channel is used to
    operate on a unit vector. The result is expressed as a complex
    number, and the spectrum of complex numbers is the Fourier
    transformed with respect to wavelength squared. The peaks of this
    transformed spectrum is taken as the rotation measure for the
    station, and used to initialise a new gaintable.

    :param gaintable: GainTable dataset to be to modelled.
    :param peak_threshold: Height of peak in the RM spectrum required for a
        rotation detection.
    :param plot_sample: Whether or not to plot a sample RM spectrum.
    "return": new GainTable dataset with model rotations.
    """
    # reference against a single station
    ref = 0

    nstations = len(gaintable.antenna)
    lambda_sq = (
        const.c.value / gaintable.frequency.data  # pylint: disable=no-member
    ) ** 2

    oversample = 5
    rm_res = 1 / oversample / (np.max(lambda_sq) - np.min(lambda_sq))
    rm_max = 1 / (lambda_sq[-2] - lambda_sq[-1])
    rm_vals = np.arange(-rm_max, rm_max, rm_res)
    uvec = np.array([1, 0])
    rm_peeks = np.zeros(nstations)
    invN = 1 / len(gaintable.frequency.data)
    invR = 1 / np.sqrt(2)
    for stn in range(nstations):
        # Reference against a single station (conj transp or inv?)
        reljones = np.einsum(
            "fpx,fqx->fpq",
            gaintable.gain.data[0, stn],
            gaintable.gain.data[0, ref].conj(),
        )
        # Normalise
        reljones /= invR * np.linalg.norm(reljones, axis=(1, 2), keepdims=True)
        # Rotate a unit vector per channel
        rvec = np.einsum("fpq,q->fp", reljones, uvec)
        # Express the results as complex numbers
        f_spec = rvec[:, 0] + 1j * rvec[:, 1]
        # Take the RM transform of the complex numbers and find the peak
        rm_spec = invN * np.einsum(
            "rf,f->r", np.exp(np.outer(-1j * rm_vals, lambda_sq)), f_spec
        )

        # Only bother if peak is significant.
        #  - real(rm_spec) is sharper, but can be affected by gain errors
        if np.max(np.abs(rm_spec)) > peak_threshold:
            rm_peeks[stn] = rm_vals[np.argmax(np.abs(rm_spec))]

        if plot_sample and stn == nstations - 1:
            xlim = 3 * np.max(np.abs(rm_peeks))
            plt.figure()
            ax = plt.subplot(111)
            ax.set_title(f"RM spectrum for station {stn}")
            ax.set_xlabel("RM (rad / m^2)")
            ax.plot(rm_vals, np.abs(rm_spec), "b", label="abs")
            ax.plot(rm_vals, np.real(rm_spec), "c", label="re")
            ax.plot(rm_vals, np.imag(rm_spec), "m", label="im")
            ax.plot(rm_peeks[stn] * np.ones(2), ax.get_ylim(), "b--")
            ax.set_xlim((-xlim, xlim))
            ax.grid()
            ax.legend()
            plt.savefig(f"{plot_path_prefix}/rm-station.png")

    modeltable = gaintable.copy(deep=True)
    for stn in range(nstations):
        d_pa = (rm_peeks[stn] - rm_peeks[0]) * lambda_sq
        modeltable.gain.data[0, stn] = np.stack(
            (np.cos(d_pa), -np.sin(d_pa), np.sin(d_pa), np.cos(d_pa)),
            axis=1,
        ).reshape(-1, 2, 2)

    return modeltable
