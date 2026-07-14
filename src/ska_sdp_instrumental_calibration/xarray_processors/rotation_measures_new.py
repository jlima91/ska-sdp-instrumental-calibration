"""Post-calibration fits."""

__all__ = [
    "model_rotations",
    "compute_rm_parameters",
    "model_rotations_ufunc",
    "get_plot_params_for_station",
]


import dask.array as da
import numpy as np
import xarray as xr
from astropy import constants as const
from scipy.optimize import curve_fit
from ska_sdp_datamodels.calibration import GainTable

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.post_processing")


def get_plot_params_for_station(
    dataset: xr.Dataset, antenna, refant, time=None
) -> dict:
    """Extract plotting parameters natively retaining lazy Dask evaluation structures.

    Parameters
    ----------
    dataset
        Dataset containing results of model rotation calculation
    antenna
        Antenna (station) number to plot
    refant
        Reference antenna used during computation
    time
        Optionally filter on solution interval

    Returns
    -------
        A dictionary with following keys and values data types:
            stn: int
            rm_vals: np.ndarray
            lambda_sq: dask.array
            rm_spec: dask.array
            rm_peak: dask.array
            rm_est: dask.array
            rm_est_refant: dask.array
            J: dask.array
            xlim: dask.array
    """
    ds = dataset.sel(time=time) if time is not None else dataset

    rm_spec = ds["rm_spec"].sel(antenna=antenna).data
    rm_peak = ds["rm_peak"].sel(antenna=antenna).data
    rm_est = ds["rm_est"].sel(antenna=antenna).data
    rm_est_refant = ds["rm_est"].sel(antenna=refant).data
    J_val = ds["J"].sel(antenna=antenna).data

    # Access the underlying dask array via .data to chain graph tasks lazily
    xlim = 10 * da.max(da.abs(ds["rm_est"].data))

    return {
        "stn": antenna,
        "rm_vals": ds.coords["resolution"].data,
        "lambda_sq": ds["lambda_sq"].data,
        "rm_spec": rm_spec,
        "rm_peak": rm_peak,
        "rm_est": rm_est,
        "rm_est_refant": rm_est_refant,
        "J": J_val,
        "xlim": xlim,
    }


def compute_rm_parameters(
    frequency: np.ndarray, oversample: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute wavelength squared and rotation measure arrays using NumPy.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency values in Hz. Shape: (freq,)
    oversample : int
        Oversampling factor.

    Returns
    -------
    lambda_sq : np.ndarray
        Wavelength squared values. Shape: (freq,)
    rm_vals : np.ndarray
        Rotation measure search grid values. Shape: (resolution,)
    """
    lambda_sq = ((const.c.value / frequency) ** 2).astype(np.float32)

    rm_res = 1 / oversample / (np.max(lambda_sq) - np.min(lambda_sq))
    rm_max = 1 / (lambda_sq[-2] - lambda_sq[-1])
    rm_max = np.ceil(rm_max / rm_res) * rm_res

    rm_vals = np.arange(
        -rm_max,
        rm_max,
        rm_res,
        dtype=np.float32,
    )
    return lambda_sq, rm_vals


def model_rotations(
    gaintable: GainTable,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
    refant: int = 0,
    oversample: int = 5,
) -> xr.Dataset:
    """
    Fit a rotation measure for each station Jones matrix.

    For each station, the approach discussed in appendix B of de Gasperin et
    al. (2019) A&A 622, A5, is used to estimate the angle of rotation in each
    Jones matrix as a function of frequency. The result is expressed as a
    spectrum of complex numbers and Fourier transformed with respect to
    wavelength squared. The peak of this RM spectrum is taken as the rotation
    measure for the station, and optionally improved upon using a secondary
    nonlinear fit to the original spectrum.

    In the current form, each matrix is first multiplied by the Hermitian
    transpose of the matrix of a reference station at the same frequency, and
    the product is normalised by the L2 norm. This will likely be updated as
    actual data are processed.

    Parameters
    ----------
    gaintable
        Bandpass calibrated gainTable dataset to be to modelled.
    peak_threshold
        Height of peak in the RM spectrum required for a
        rotation detection.
    refine_fit
        Whether or not to refine the RM spectrum peak locations
        with a nonlinear optimisation of the station RM values.
    refant
        Reference antenna
    oversample
        Oversampling value used in the rotation
        calculation. This determines the resolution of phasor.
        Note that setting this value to some higher
        integer may result in high memory usage.

    Returns
    -------
        A Dataset
    """
    gaintable = gaintable.chunk(time=1, antenna=1, frequency=-1)
    gaintable_refant = gaintable.isel(antenna=refant, drop=True)

    lambda_sq, rm_vals_coords = compute_rm_parameters(
        gaintable["frequency"].values, oversample
    )

    n_time = gaintable.sizes["time"]
    n_ant = gaintable.sizes["antenna"]
    n_res = rm_vals_coords.size
    n_freq = gaintable.sizes["frequency"]
    n_rec1 = gaintable.sizes["receptor1"]
    n_rec2 = gaintable.sizes["receptor2"]

    time_chunks = gaintable.chunks["time"]
    ant_chunks = gaintable.chunks["antenna"]

    template = xr.Dataset(
        data_vars=dict(
            lambda_sq=(["frequency"], lambda_sq),
            rm_spec=(
                ["time", "antenna", "resolution"],
                da.empty(
                    (n_time, n_ant, n_res),
                    chunks=(time_chunks, ant_chunks, n_res),
                    dtype=np.complex128,
                ),
            ),
            rm_est=(
                ["time", "antenna"],
                da.empty(
                    (n_time, n_ant),
                    chunks=(time_chunks, ant_chunks),
                    dtype=np.float64,
                ),
            ),
            rm_peak=(
                ["time", "antenna"],
                da.empty(
                    (n_time, n_ant),
                    chunks=(time_chunks, ant_chunks),
                    dtype=np.float64,
                ),
            ),
            const_rot=(
                ["time", "antenna"],
                da.empty(
                    (n_time, n_ant),
                    chunks=(time_chunks, ant_chunks),
                    dtype=np.float64,
                ),
            ),
            J=(
                ["time", "antenna", "frequency", "receptor1", "receptor2"],
                da.empty(
                    (n_time, n_ant, n_freq, n_rec1, n_rec2),
                    chunks=(time_chunks, ant_chunks, n_freq, n_rec1, n_rec2),
                    dtype=gaintable["gain"].dtype,
                ),
            ),
        ),
        coords=dict(
            time=gaintable.coords["time"],
            antenna=gaintable.coords["antenna"],
            frequency=gaintable.coords["frequency"],
            resolution=rm_vals_coords,
            receptor1=gaintable.coords["receptor1"],
            receptor2=gaintable.coords["receptor2"],
        ),
    )

    result_ds = template.map_blocks(
        _model_rotation_block_,
        args=(gaintable, gaintable_refant),
        kwargs=dict(
            peak_threshold=peak_threshold,
            refine_fit=refine_fit,
        ),
        template=template,
    )

    return result_ds


def _model_rotation_block_(
    template_block: xr.Dataset,
    gaintable: xr.Dataset,
    gaintable_refant: xr.Dataset,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
) -> xr.Dataset:
    """
    Processes one template block (time x antenna) at a time, converting
    xarray inputs to pure numpy for computation and wrapping the results back.
    """
    # Extract coordinates directly from the template block to guarantee absolute consistency
    rm_vals = template_block.coords["resolution"].values
    lambda_sq = template_block["lambda_sq"].values

    time_coord = template_block.coords["time"]
    antenna_coord = template_block.coords["antenna"]
    frequency_coord = template_block.coords["frequency"]

    # Drop 1-element time and antenna dimensions for core block extraction
    gaintable_squeezed = gaintable.squeeze(dim=["time", "antenna"])
    gaintable_refant_squeezed = gaintable_refant.squeeze(dim="time")

    # Extract raw NumPy arrays from the squeezed Xarray Datasets
    gain = gaintable_squeezed["gain"].values
    weight = gaintable_squeezed["weight"].values
    gain_refant = gaintable_refant_squeezed["gain"].values
    weight_refant = gaintable_refant_squeezed["weight"].values

    J, rm_spec, rm_est, rm_peak, const_rot = model_rotations_ufunc(
        gain=gain,
        weight=weight,
        gain_refant=gain_refant,
        weight_refant=weight_refant,
        rm_vals=rm_vals,
        lambda_sq=lambda_sq,
        peak_threshold=peak_threshold,
        refine_fit=refine_fit,
    )

    block_ds = xr.Dataset(
        data_vars=dict(
            lambda_sq=(["frequency"], lambda_sq),
            rm_spec=(
                ["time", "antenna", "resolution"],
                rm_spec[np.newaxis, np.newaxis, :],
            ),
            rm_est=(
                ["time", "antenna"],
                np.array([[rm_est]], dtype=np.float64),
            ),
            rm_peak=(
                ["time", "antenna"],
                np.array([[rm_peak]], dtype=np.float64),
            ),
            const_rot=(
                ["time", "antenna"],
                np.array([[const_rot]], dtype=np.float64),
            ),
            J=(
                ["time", "antenna", "frequency", "receptor1", "receptor2"],
                J[np.newaxis, np.newaxis, ...],
            ),
        ),
        coords=dict(
            time=time_coord,
            antenna=antenna_coord,
            frequency=frequency_coord,
            resolution=rm_vals,
            receptor1=template_block["receptor1"],
            receptor2=template_block["receptor2"],
        ),
    )

    return block_ds


def model_rotations_ufunc(
    gain: np.ndarray,
    weight: np.ndarray,
    gain_refant: np.ndarray,
    weight_refant: np.ndarray,
    rm_vals: np.ndarray = None,
    lambda_sq: np.ndarray = None,
    frequency: np.ndarray = None,
    oversample: int = 5,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Calculate model rotations for a given antenna and solution interval (time)
    using pure NumPy arrays. This can be treated as a ufunc, and can be used to broadcast computations
    across stations and time.

    Parameters
    ----------
    gain
        Calibrated gain matrix array for a single station block. Shape: (freq, rec1, rec2)
    weight
        Weight matrix array for a single station block. Shape: (freq, rec1, rec2)
    gain_refant
        Calibrated gain matrix array for the reference station. Shape: (freq, rec1, rec2)
    weight_refant
        Weight matrix array for the reference station. Shape: (freq, rec1, rec2)
    rm_vals
        Pre-computed rotation measure search grid values. Shape: (resolution,).
    lambda_sq
        Pre-computed wavelength squared values. Shape: (freq,).
    frequency
        Frequency values in Hz. Only utilized if rm_vals/lambda_sq must be computed. Shape: (freq,).
    oversample
        Oversampling value used if parameters must be generated on the fly. Only utilized if rm_vals/lambda_sq must be computed.
    peak_threshold
        Height of peak in the RM spectrum required for a rotation detection.
    refine_fit
        Whether or not to refine the RM spectrum peak locations with a nonlinear optimisation.

    Returns
    -------
        J
            Rotation matrix w.r.t. reference antenna. Shape: (freq, rec1, rec2)
        rm_spec
            Array of complex RM spectrum values. Shape: (resolution,)
        rm_est
            Final optimized rotation measure estimation.
        rm_peak
            Initial peak rotation measure estimation.
        const_rot
            Constant phase rotation tracking value.
    """
    if rm_vals is None or lambda_sq is None:
        if frequency is None:
            raise ValueError(
                "frequency must be provided if rm_vals or lambda_sq are None"
            )
        lambda_sq, rm_vals = compute_rm_parameters(frequency, oversample)

    J = np.einsum(
        "fpx,fqx->fpq",
        gain_refant.conj(),
        gain,
        dtype=gain.dtype,
    )  # Shape: (freq,rec1,rec2)

    norms = np.linalg.norm(
        J, axis=(1, 2), keepdims=True
    )  # Shape: (freq,rec1,rec2)
    mask = get_stn_masks(weight, weight_refant)  # Shape: (freq,)
    mask = mask & (norms[:, 0, 0] > 0)  # Shape: (freq,)

    J = update_jones_with_masks(J, mask, norms)  # Shape: (freq,rec1,rec2)
    phi_raw = calculate_phi_raw(J)  # Shape: (freq,)

    rm_spec = get_rm_spec(
        phi_raw, mask, rm_vals, lambda_sq
    )  # Shape: (resolution,)
    abs_rm_spec = np.abs(rm_spec)

    rm_est = (
        rm_vals[np.argmax(abs_rm_spec)]
        if np.max(abs_rm_spec) > peak_threshold
        else 0.0
    )

    rm_peak = rm_est
    const_rot = 0.0

    if refine_fit:
        rm_est, const_rot = fit_curve(lambda_sq, phi_raw, rm_peak)

    return J, rm_spec, rm_est, rm_peak, const_rot


def calculate_phi_raw(jones: np.ndarray) -> np.ndarray[float]:
    """
    Parameters
    ----------
    jones
        Jones array for given antenna (freq, rec1, rec2)

    Returns
    -------
        Raw phase angle (freq). float
    """
    co_sum = jones[:, 0, 0] + jones[:, 1, 1]
    cross_diff = 1j * (jones[:, 0, 1] - jones[:, 1, 0])
    return 0.5 * (
        np.unwrap(np.angle(co_sum + cross_diff))
        - np.unwrap(np.angle(co_sum - cross_diff))
    )


def update_jones_with_masks(
    jones: np.ndarray, mask: np.ndarray, norms: np.ndarray
):
    """
    Update the Jones array for mask values
    Note that this function mutates jones array in place

    Parameters
    ----------
    jones
        Jones array which is mutated (freq, rec1, rec2)
    mask
        Mask for stations. (freq)
    norms
        Normalization array. (freq, rec1, rec2)

    Returns
    -------
        Array of updated jones values. Same as input jones.
    """
    jones[mask, :, :] *= np.sqrt(2) / norms[mask, :, :]
    return jones


def get_stn_masks(weight_ant: np.ndarray, weight_refant: np.ndarray):
    """
    Gets station masks for given antenna

    Parameters
    ----------
    weight_ant
        Weight of current antenna. (freq, rec1, rec2)
    weight_refant
        Reference antenna Weight. (freq, rec1, rec2)

    Returns
    -------
    Mask for current antenna. (freq)
    """
    if np.all(weight_refant[:, 0, 1] == 0) & np.all(
        weight_refant[:, 1, 0] == 0
    ):
        return (
            (weight_ant[:, 0, 0] > 0)
            & (weight_ant[:, 1, 1] > 0)
            & (weight_refant[:, 0, 0] > 0)
            & (weight_refant[:, 1, 1] > 0)
        )
    return np.all(weight_ant > 0, axis=(1, 2)) & np.all(
        weight_refant > 0, axis=(1, 2)
    )


def get_rm_spec(
    phi_raw: np.ndarray,
    mask: np.ndarray,
    rm_vals: np.ndarray,
    lambda_sq: np.ndarray,
    chunk_size=4096,
) -> np.ndarray[complex]:
    """
    Calculate RM spec for current antenna

    Parameters
    ----------
    phi_raw
        Phi raw value. Shape: ``(freq,)``. float
    mask
        Mask. Shape: ``(freq,)``. bool.
    rm_vals
        Rotation measure values. ``(resolution,)``. float
    lambda_sq
        Wavelength squared. ``(freq,)``. float
    chunk_size
        Max resolution to compute at a time. Used for memory efficiency.

    Returns
    -------
        Array of RM spec. Shape: ``(resolution)``. complex
    """
    resolution = rm_vals.shape[0]

    phi_exp = np.exp(1j * phi_raw)
    masked_phi = np.where(mask, phi_exp, 0j)
    counts = mask.sum(dtype=np.float32)

    result = np.empty((resolution,), dtype=np.complex128)

    # Process the resolution axis in chunks
    for i in range(0, resolution, chunk_size):
        end = min(i + chunk_size, resolution)
        rm_chunk = rm_vals[i:end]

        # Intermediate array shape: (nchannels, chunk_size)
        # Instead of allocating (nchannels, resolution) all at once
        phasor_chunk = np.exp(
            -1j * lambda_sq[:, np.newaxis] * rm_chunk[np.newaxis, :]
        )
        result[i:end] = masked_phi @ phasor_chunk

    return result / counts


def fit_curve(
    lambda_sq: np.ndarray, phi_raw: np.ndarray, rm_est: float
) -> tuple[float, float]:
    """
    Fits the curve

    Parameters
    ----------
    lambda_sq
        Wavelength square. (freq,) float
    phi_raw
        Raw phase angle (freq,) float
    rm_est
        Initial rm estimate for current station.
        Shape: ()

    Returns
    -------
        (rm_est, const_rot)
        New rotation measure estimate and constant rotation values
        post curve fitting
    """
    exp_stack = np.hstack([np.cos(phi_raw), np.sin(phi_raw)])

    popt, _ = curve_fit(
        lambda wl2, rm, phi0: np.hstack(
            (np.cos(wl2 * rm + phi0), np.sin(wl2 * rm + phi0))
        ),
        lambda_sq,
        exp_stack,
        p0=[rm_est, 0],
    )
    return float(popt[0]), float(popt[1])
