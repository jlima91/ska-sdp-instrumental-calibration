"""Post-calibration fits."""

__all__ = ["model_rotations"]


import dask
import dask.array as da
import numpy as np
import xarray as xr
from astropy import constants as const
from scipy.optimize import curve_fit

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.post_processing")

# Some opinionated chunksizes
RESOLUTION_CHUNK_SIZE = 2**11
FREQ_CHUNK_SIZE = -1
STATION_CHUNK_SIZE = 1


@dask.delayed
def delayed_einsum(*args, **kwargs):
    """
    Sometimes einsum just blows up tasks
    This will force compute on a single worker
    """
    return np.einsum(*args, **kwargs)


class ModelRotationData:
    """
    Create Model Rotation Data

    Parameters
    ----------
    gaintable: Gaintable dataset
        Gaintable.
    refant: int
        Reference antenna.
    oversample: int, default: 5
        Oversampling value used in the rotation
        calculatiosn. Note that setting this value to some higher
        integer may result in high memory usage.
    """

    def __init__(self, gaintable, refant, oversample=5):
        if gaintable.gain.shape[3] != 2 or gaintable.gain.shape[4] != 2:
            raise ValueError("gaintable must contain Jones matrices")

        self.gaintable = gaintable
        self.refant = refant
        self.nstations = len(gaintable.antenna)
        self.nfreq = len(gaintable.frequency)

        lambda_sq_npa = (
            (
                const.c.value  # pylint: disable=no-member
                / gaintable.frequency.data
            )
            ** 2
        ).astype(np.float32)

        self.rm_res = (
            1 / oversample / (np.max(lambda_sq_npa) - np.min(lambda_sq_npa))
        )
        self.rm_max = 1 / (lambda_sq_npa[-2] - lambda_sq_npa[-1])
        self.rm_max = np.ceil(self.rm_max / self.rm_res) * self.rm_res

        self.rm_vals = da.arange(
            -self.rm_max,
            self.rm_max,
            self.rm_res,
            dtype=np.float32,
            chunks=(RESOLUTION_CHUNK_SIZE),
        )
        self.lambda_sq = da.from_array(lambda_sq_npa, chunks=(FREQ_CHUNK_SIZE))
        self.rm_spec = None

        self.rm_est = da.zeros(self.nstations, dtype=np.float32)
        self.rm_peak = da.zeros(self.nstations, dtype=np.float32)
        self.const_rot = da.zeros(self.nstations, dtype=np.float32)
        self.J = da.from_delayed(
            delayed_einsum(
                "fpx,sfqx->sfpq",
                gaintable.gain[0, refant].conj(),
                gaintable.gain[0, :],
                dtype=np.complex64,
            ),
            (gaintable.gain.shape[1:]),
            np.complex64,
            name="rm_J",
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
    gaintable: xr.Dataset,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
    refant: int = 0,
    oversample: int = 5,
):
    """
    Performs Model Rotations

    Parameters
    ----------
        gaintable: Gaintable dataset
            Gaintable.
        peak_threshold: float
            Peak threshold.
        refine_fit: bool
            Refine the fit.
        refant: int, default: 0
            Reference antenna.
        oversample: int, default: 5
            Oversampling value used in the rotation
            calculatiosn. Note that setting this value to some higher
            integer may result in high memory usage.

    Returns
    -------
        rotations: ModelRotation obj.
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

    # Some opinionated rechunking
    phi_raw = phi_raw.rechunk((STATION_CHUNK_SIZE, FREQ_CHUNK_SIZE))
    mask = mask.rechunk((STATION_CHUNK_SIZE, FREQ_CHUNK_SIZE))

    phasor = da.exp(
        np.outer(-1j * rotations.rm_vals, rotations.lambda_sq),
    )

    rotations.rm_spec = get_rm_spec(
        phi_raw,
        mask,
        phasor,
    )

    abs_rm_spec = da.abs(rotations.rm_spec)

    rotations.rm_est = da.where(
        da.max(abs_rm_spec, axis=1) > peak_threshold,
        rotations.rm_vals[da.argmax(abs_rm_spec, axis=1)],
        0,
    )

    rotations.rm_peak = rotations.rm_est

    if refine_fit:
        exp_stack = da.hstack((da.cos(phi_raw), da.sin(phi_raw)))
        fit_rm = fit_curve(
            rotations.lambda_sq,
            exp_stack,
            rotations.rm_peak,
            rotations.nstations,
        )

        rotations.rm_est = fit_rm[0]
        rotations.rm_const = fit_rm[1]

    # Output expects all station values to be together
    rotations.rm_est = rotations.rm_est.rechunk(-1)

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


def get_rm_spec(phi_raw, mask, phasor):
    """
    Gets RM spec.
    Data can be chunked in ``nstations`` and ``resolution`` dimensions.

    Parameters
    ----------
        phi_raw: np.array
            Phi raw value. Shape: ``(nstations, nchannels)``
        mask: np.array
            Mask. Shape: ``(nstations, nchannels)``
        phasor: np.array
            Phasor. Shape: ``(resolution, nchannels)``

    Returns
    -------
    np.ndarray
        Array of RM spec. Shape: ``(nstations, resolution)``
        Will have same number of chunks as they are in input arrays.
    """
    return da.stack(
        [
            (
                da.einsum(
                    "rf,f->r",
                    phasor[:, mask[stn]],
                    da.exp(1j * phi_raw[stn, mask[stn]]),
                    dtype=np.complex64,
                )
                / da.sum(mask[stn], dtype=np.float32)
            )
            for stn in range(mask.shape[0])
        ]
    )


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
    result = [
        da.from_delayed(
            get_optimised_parameters(
                lambda_sq,
                exp_stack[stn],
                rm_est[stn],
            ),
            (2,),
            np.float32,
        )
        for stn in range(nstations)
    ]

    return da.stack(result).T


@dask.delayed
def get_optimised_parameters(lambda_sq, exp_stack, rm_est):
    """
    Calculates optimal parameters

    Returns
    -------
    np.ndarray
        1D array of size 2.
    """
    return curve_fit(
        lambda wl2, rm, phi0: np.hstack(
            (np.cos(wl2 * rm + phi0), np.sin(wl2 * rm + phi0))
        ),
        lambda_sq,
        exp_stack,
        [rm_est, 0],
    )[0]
