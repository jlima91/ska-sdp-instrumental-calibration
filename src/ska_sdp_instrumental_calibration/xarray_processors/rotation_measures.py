"""Post-calibration fits."""

__all__ = ["model_rotations"]


import dask
import dask.array as da
import numpy as np
from astropy import constants as const
from scipy.optimize import curve_fit
from ska_sdp_datamodels.calibration import GainTable

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.post_processing")


class ModelRotationData:
    """
    Create Model Rotation Data

    Parameters
    ----------
    gaintable
        Calibrated gaintable
    refant
        Reference antenna.
    oversample
        Oversampling value used in the rotation
        calculatiosn. Note that setting this value to some higher
        integer may result in high memory usage.
    """

    def __init__(self, gaintable: GainTable, refant: int, oversample: int = 5):
        if gaintable.gain.shape[3] != 2 or gaintable.gain.shape[4] != 2:
            raise ValueError("gaintable must contain Jones matrices")

        self.gaintable = gaintable
        self.refant = refant
        self.nstations = len(gaintable.antenna)
        self.nfreq = len(gaintable.frequency)
        self.lambda_sq = (
            (
                const.c.value  # pylint: disable=no-member
                / gaintable.frequency.data
            )
            ** 2
        ).astype(np.float32)

        self.rm_res = (
            1 / oversample / (da.max(self.lambda_sq) - da.min(self.lambda_sq))
        )
        self.rm_max = 1 / (self.lambda_sq[-2] - self.lambda_sq[-1])
        self.rm_max = da.ceil(self.rm_max / self.rm_res) * self.rm_res
        self.rm_vals = da.arange(
            -self.rm_max, self.rm_max, self.rm_res, dtype=np.float32
        )
        self.phasor = da.exp(
            da.einsum("i,j->ij", -1j * self.rm_vals, self.lambda_sq)
        )
        self.rm_spec = None

        self.rm_est = da.zeros(self.nstations, dtype=np.float32)
        self.rm_peak = da.zeros(self.nstations, dtype=np.float32)
        self.const_rot = da.zeros(self.nstations, dtype=np.float32)
        self.J = da.einsum(
            "fpx,sfqx->sfpq",
            gaintable.gain[0, refant].conj(),
            gaintable.gain[0, :],
            dtype=np.complex64,
        )

    def get_plot_params_for_station(self, stn=None):
        """
        Getter for plot params for any particular station.

        Parameters
        ----------
            stn: int
                Station number.
        Returns
        -------
            rm_vals: dask.array
                rm value array.
            rm_spec: dask.array
                rm spec array.
            rm_peak: dask.array
                rm peak array.
            rm_est: dask.array
                rm estimate array.
            rm_est_refant: dask.array
                rm estimate of refant.
            J: dask.array
                Jones array.
            lambda_sq: dask.array
                lambda square array.
            xlim: dask.array
                x-limit array.
            stn: int
                Station number.
        """
        stn = stn if stn is not None else len(self.gaintable.antenna) - 1

        return {
            "rm_vals": self.rm_vals,
            "rm_spec": self.rm_spec[stn],
            "rm_peak": self.rm_peak[stn],
            "rm_est": self.rm_est[stn],
            "rm_est_refant": self.rm_est[self.refant],
            "J": self.J[stn],
            "lambda_sq": self.lambda_sq,
            "xlim": 10 * da.max(da.abs(self.rm_est)),
            "stn": stn,
        }


def model_rotations(
    gaintable: GainTable,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
    refant: int = 0,
    oversample: int = 5,
) -> ModelRotationData:
    """
    Performs Model Rotations

    Parameters
    ----------
        gaintable
            Bandpass calibrated gaintable
        peak_threshold
            Peak threshold.
        refine_fit
            Refine the fit.
        refant
            Reference antenna.
        oversample
            Oversampling value used in the rotation
            calculatiosn. Note that setting this value to some higher
            integer may result in high memory usage.

    Returns
    -------
        rotations.
    """

    rotations = ModelRotationData(gaintable, refant, oversample)

    norms = da.linalg.norm(rotations.J, axis=(2, 3), keepdims=True)
    mask = da.from_delayed(
        get_stn_masks(gaintable.weight, refant),
        (
            rotations.nstations,
            rotations.nfreq,
        ),
        bool,
    )

    mask = mask & (norms[:, :, 0, 0] > 0)

    rotations.J = da.from_delayed(
        update_jones_with_masks(rotations.J, mask, norms, rotations.nstations),
        rotations.J.shape,
        rotations.J.dtype,
    )

    phi_raw = da.from_delayed(
        calculate_phi_raw(rotations.J),
        (rotations.nstations, rotations.nfreq),
        np.float32,
    )

    rotations.rm_spec = da.from_delayed(
        get_rm_spec(phi_raw, rotations.phasor, mask, rotations.nstations),
        (rotations.nstations, rotations.phasor.shape[0]),
        np.float32,
    )

    rotations.rm_est = da.where(
        da.max(da.abs(rotations.rm_spec), axis=1) > peak_threshold,
        rotations.rm_vals[da.argmax(da.abs(rotations.rm_spec), axis=1)],
        0,
    )

    rotations.rm_peak = rotations.rm_est

    if refine_fit:
        exp_stack = da.hstack((da.cos(phi_raw), da.sin(phi_raw)))
        fit_rm = da.from_delayed(
            fit_curve(
                rotations.lambda_sq,
                exp_stack,
                rotations.rm_peak,
                rotations.nstations,
            ),
            (2, rotations.nstations),
            np.float32,
        )
        rotations.rm_est = fit_rm[0]
        rotations.rm_const = fit_rm[1]

    return rotations


@dask.delayed
def calculate_phi_raw(jones):
    co_sum = jones[:, :, 0, 0] + jones[:, :, 1, 1]
    cross_diff = 1j * (jones[:, :, 0, 1] - jones[:, :, 1, 0])
    return 0.5 * (
        np.unwrap(np.angle(co_sum + cross_diff))
        - np.unwrap(np.angle(co_sum - cross_diff))
    )


@dask.delayed
def update_jones_with_masks(jones, mask, norms, nstations):
    """
    Update the Jones array for mask values
    Parameters
    ----------
        jones: np.array
            Jones array.
        mask: np.array
            Mask for stations.
        norms: np.array
            Normalization array.
        nstations: int
            Number of stations.
    Returns
    -------
        Array of updated jones values.
    """
    for stn in range(nstations):
        jones[stn, mask[stn], :, :] *= np.sqrt(2) / norms[stn, mask[stn], :, :]

    return jones


@dask.delayed
def get_stn_masks(weight, refant):
    """
    Gets station masks.

    Parameters
    ----------
        weight: np.array
            Weight.
        refant: int
            Reference antenna.
     Returns
     -------
        Array of masks for stations.
    """
    if np.all(weight[0, refant, :, 0, 1] == 0) & np.all(
        weight[0, refant, :, 1, 0] == 0
    ):
        return (
            (weight[0, :, :, 0, 0] > 0)
            & (weight[0, :, :, 1, 1] > 0)
            & (weight[0, refant, :, 0, 0] > 0)
            & (weight[0, refant, :, 1, 1] > 0)
        )

    return np.all(weight[0, :] > 0, axis=(2, 3)) & np.all(
        weight[0, refant] > 0, axis=(1, 2)
    )


@dask.delayed
def get_rm_spec(phi_raw, phasor, mask, nstations):
    """
    Gets RM spec

    Parameters
    ----------
        phi_raw: np.array
            Phi raw value
        phasor: np.array
            Phasor
        mask: np.array
            Mask
        nstations: int
            Number of stations.
    Returns
    -------
        Array of RM spec.
    """
    return np.array(
        [
            (
                1
                / sum(mask[stn])
                * np.einsum(
                    "rf,f->r",
                    phasor[:, mask[stn]],
                    np.exp(1j * phi_raw[stn, mask[stn]]),
                )
            )
            for stn in range(nstations)
        ]
    )


@dask.delayed
def fit_curve(lambda_sq, exp_stack, rm_est, nstations):
    """
    Fits the curve

    Parameters
    ----------
        lambda_sq: np.array
            Lambda square
        exp_stack: np.array
            exp stack
        rm_est: np.array
            rm estimate
        nstations: int
            Number of stations
    Returns
    -------
        Array after fitting the curve.
    """
    return np.array(
        [
            (popt[0], popt[1])
            for popt, _ in [
                curve_fit(
                    lambda wl2, rm, phi0: np.hstack(
                        (np.cos(wl2 * rm + phi0), np.sin(wl2 * rm + phi0))
                    ),
                    lambda_sq,
                    exp_stack[stn],
                    p0=[rm_est[stn], 0],
                )
                for stn in range(nstations)
            ]
        ]
    ).T
