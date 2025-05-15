"""Post-calibration fits."""

__all__ = ["model_rotations"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr
from astropy import constants as const
from scipy.optimize import curve_fit

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.post_processing")


def rm_func(
    wl2: npt.NDArray[float], rm: float, phi0: float
) -> npt.NDArray[float]:
    """Function to use in the curve_fit optimisations of model_rotations.

    Express the rotation measure at a set of wavelength squared values as the
    complex exponential, exp(i * (wl2 * rm + phi0)), separated into real and
    imaginary parts.

    :param wl2: Wavelength squared values at which to evaluate the function.
    :param rm: Rotation measure.
    :param phi0: Constant rotation angle.
    :return: cosine spectrum and sine spectrum stacked into a single vector.
    """
    return np.hstack((np.cos(wl2 * rm + phi0), np.sin(wl2 * rm + phi0)))


def model_rotations(
    gaintable: xr.Dataset,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
    plot_sample: bool = False,
    plot_path_prefix: str = "./",
) -> npt.NDArray[float]:
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
    :param refine_fit: Whether or not to refine the RM spectrum peak locations
        with a nonlinear optimisation of the station RM values.
    :param plot_sample: Whether or not to plot a sample RM spectrum.
    :return: Estimated station RM values.
    """
    if gaintable.gain.shape[3] != 2 or gaintable.gain.shape[4] != 2:
        raise ValueError("gaintable must contain Jones matrices")

    # Set reference station for rotations
    ref = 0

    # Set constants
    nstations = len(gaintable.antenna)
    lambda_sq = (
        const.c.value / gaintable.frequency.data  # pylint: disable=no-member
    ) ** 2

    # Set RM spectrum parameters
    oversample = 99
    rm_res = 1 / oversample / (np.max(lambda_sq) - np.min(lambda_sq))
    rm_max = 1 / (lambda_sq[-2] - lambda_sq[-1])
    rm_max = np.ceil(rm_max / rm_res) * rm_res
    rm_vals = np.arange(-rm_max, rm_max, rm_res)
    phasor = np.exp(np.outer(-1j * rm_vals, lambda_sq))

    rm_est = np.zeros(nstations)
    const_rot = np.zeros(nstations)

    # Set RM intergral weights. Don't want variable weights across a matrix,
    # so for now just set a flag if any matrix element has zero weight.
    # Some solvers may set a diagonal weight matrix, so test.
    diag_weights = np.all(
        gaintable.weight.data[0, ref, :, 0, 1] == 0
    ) & np.all(gaintable.weight.data[0, ref, :, 1, 0] == 0)
    if diag_weights:
        # Just check the cross pol weights
        ref_mask = (gaintable.weight.data[0, ref, :, 0, 0] > 0) & (
            gaintable.weight.data[0, ref, :, 1, 1] > 0
        )
    else:
        # Check all polarisation weights
        ref_mask = np.all(gaintable.weight.data[0, ref] > 0, axis=(1, 2))

    for stn in range(nstations):
        # Reference against a single station
        #  - should this be the conj transpose or the inverse?
        #  - is it better to ref on the RHS or LHS?
        J = np.einsum(
            "fpx,fqx->fpq",
            # gaintable.gain.data[0, stn],
            # gaintable.gain.data[0, ref].conj(),
            gaintable.gain.data[0, ref].conj(),
            gaintable.gain.data[0, stn],
        )
        # Normalise
        J *= np.sqrt(2) / np.linalg.norm(J, axis=(1, 2), keepdims=True)

        # Extract the rotation angle from each Jones matrix
        co_sum = J[:, 0, 0] + J[:, 1, 1]
        cross_diff = 1j * (J[:, 0, 1] - J[:, 1, 0])
        phi_raw = 0.5 * (
            #  - changed order when the ref side was switched for J...
            # np.unwrap(np.angle(co_sum - cross_diff))
            # - np.unwrap(np.angle(co_sum + cross_diff))
            np.unwrap(np.angle(co_sum + cross_diff))
            - np.unwrap(np.angle(co_sum - cross_diff))
        )

        # Take the RM transform of the rotations and find the peak
        if diag_weights:
            mask = (
                (gaintable.weight.data[0, stn, :, 0, 0] > 0)
                & (gaintable.weight.data[0, stn, :, 1, 1] > 0)
                & ref_mask
            )
        else:
            mask = (
                np.all(gaintable.weight.data[0, stn] > 0, axis=(1, 2))
                & ref_mask
            )
        rm_spec = (
            1
            / sum(mask)
            * np.einsum("rf,f->r", phasor[:, mask], np.exp(1j * phi_raw[mask]))
        )

        rm_peek = 0
        # If there isn't a clear peak, leave rm_peak set to zero
        if np.max(np.abs(rm_spec)) > peak_threshold:
            # the real peak is sharper, but can be shifted by other factors
            rm_peek = rm_vals[np.argmax(np.abs(rm_spec))]

        if refine_fit:
            # Refine the RM estimate with a non-linear fit
            #  - convert the extracted rotation angles to the rm_func format
            exp_stack = np.hstack((np.cos(phi_raw), np.sin(phi_raw)))
            #  - find updated values for the RM and a constant rotation (phi0)
            popt, _ = curve_fit(rm_func, lambda_sq, exp_stack, p0=[rm_peek, 0])
            rm_est[stn] = popt[0]
            const_rot[stn] = popt[1]
        else:
            rm_est[stn] = rm_peek
            const_rot[stn] = 0

        if plot_sample and stn == nstations - 1:

            plt.figure(figsize=(14, 12))

            x = gaintable.frequency.data / 1e6

            ax = plt.subplot(311)
            ax.plot(rm_vals, np.abs(rm_spec), "b", label="abs")
            ax.plot(rm_vals, np.real(rm_spec), "c", label="re")
            ax.plot(rm_vals, np.imag(rm_spec), "m", label="im")
            ax.plot(rm_peek * np.ones(2), ax.get_ylim(), "c-")
            ax.plot(rm_est[stn] * np.ones(2), ax.get_ylim(), "b--")
            xlim = 10 * np.max(np.abs(rm_est))
            ax.set_xlim((-xlim, xlim))
            ax.set_title(f"RM spectrum. Peak = {rm_est[stn]:.3f} (rad / m^2)")
            ax.set_xlabel("RM (rad / m^2)")
            ax.grid()
            ax.legend()

            ax = plt.subplot(323)
            for pol in range(4):
                p = pol // 2
                q = pol % 2
                ax.plot(x, np.real(J[:, p, q]), f"C{pol}", label=f"J{p}{q}")
                ax.plot(x, np.imag(J[:, p // 2, p % 2]), f"C{pol}--")
            ax.set_title(f"Bandpass Jones for station {stn} (re: -, im: --)")
            ax.grid()
            ax.legend()

            # Why does this rotation need to be transposed?
            #  - check all exponent signs and matrix directions...
            #  - it changes when the ref side is switched for J...
            ax = plt.subplot(324)
            d_pa = (rm_est[stn] - rm_est[0]) * lambda_sq
            R = np.stack(
                (np.cos(d_pa), np.sin(d_pa), -np.sin(d_pa), np.cos(d_pa)),
                axis=1,
            ).reshape(-1, 2, 2)
            for p in range(4):
                ax.plot(x, np.real(R[:, p // 2, p % 2]), f"C{p}")
                ax.plot(x, np.imag(R[:, p // 2, p % 2]), f"C{p}--")
            ax.set_title("RM rotation matrices")
            ax.grid()

            ax = plt.subplot(325)
            B = J @ np.linalg.inv(R[..., :, :])
            for p in range(4):
                ax.plot(x, np.real(B[:, p // 2, p % 2]), f"C{p}")
                ax.plot(x, np.imag(B[:, p // 2, p % 2]), f"C{p}--")
            ax.set_title("De-rotated bandpass Jones (re: -, im: --)")
            ax.set_xlabel("Frequency (MHz)")
            ax.grid()

            ax = plt.subplot(326)
            B = J @ np.linalg.inv(R[..., :, :])
            for p in [0, 3]:
                ax.plot(x, np.abs(B[:, p // 2, p % 2]), f"C{p}")
                ax.plot(x, np.angle(B[:, p // 2, p % 2]), f"C{p}--")
            ax.set_title("De-rotated bandpass Jones (abs: -, angle: --)")
            ax.set_xlabel("Frequency (MHz)")
            ax.grid()

            plt.savefig(f"{plot_path_prefix}/rm-station.png")

    return rm_est
