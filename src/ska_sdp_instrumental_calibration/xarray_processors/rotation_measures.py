"""Post-calibration fits."""

import logging

import dask.array as da
import numpy as np
import ppigrf
import xarray as xr
from astropy import constants as const
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.optimize import minimize_scalar
from ska_sdp_datamodels.visibility import Visibility

from ..data_managers.schema.calibration_set import GainCalibrationSetXds
from ..data_managers.schema.polarization import jones_to_pol, pol_to_jones
from .ionosphere_solvers import sample_external_tec

__all__ = [
    "RotationMeasureData",
    "get_plot_params_for_station",
    "model_rotations",
    "rm_from_external_tec",
]

logger = logging.getLogger(__name__)


class RotationMeasureData(xr.Dataset):
    """A specialized xarray Dataset subclass representing structural
    Rotation Measure (RM) modeling outputs.

    **Coordinates**

    - time: time centroids of solutions, in seconds elapsed since the MJD
      reference epoch, ``[ntimes]``.
    - antenna_name: antenna names, ``[nants]``.
    - frequency: center frequencies of the observations in Hz, ``[nchan]``.
    - resolution: Faraday depth search grid space values in rad/m^2,
      ``[nres]``.
    - polarization: polarization labels of the flattened Jones matrices,
      ``[npol]``. Most likely ``['XX', 'XY', 'YX', 'YY']``
      or ``['I']``.

    **Data variables**

    - lambda_sq: wavelength squared values calculated across the
      frequency axis, real-valued ``[nchan]``.
    - rm_spec: Faraday dispersion function complex spectrum profiles,
      complex-valued ``[ntimes, nants, nres]``.
    - rm_est: final non-linear optimized rotation measure parameter
      estimations, real-valued ``[ntimes, nants]``.
    - rm_peak: Peak rotation measure absolute maxima found within the search
      space, real-valued ``[ntimes, nants]``.
    - const_rot: constant intrinsic phase offset calculated post
      curve-fitting optimization, real-valued ``[ntimes, nants]``.
    - J: target Jones matrices corrected relative to the designated
      reference antenna, complex-valued
      ``[ntimes, nants, nchan, npol]``.

    **Attributes**

    - data_model: name of this class, used internally for saving to /
      loading from files.

    Here is an example::

        <xarray.RotationMeasureData>
        Dimensions:  (time: 1, antenna_name: 115, frequency: 256,
                      resolution: 1024, polarization: 4)
        Coordinates:
          * time        (time) float64 5.085e+09
          * antenna_name (antenna_name) <U8 'C1' 'C2' ... 'S16-6'
          * frequency   (frequency) float64 ...
          * resolution  (resolution) float32 ...
          * polarization (polarization)  'XX' 'XY' 'YX' 'YY'
        Data variables:
            lambda_sq   (frequency) float32 ...
            rm_spec     (time, antenna_name, resolution) complex128 ...
            rm_est      (time, antenna_name) float64 ...
            rm_peak     (time, antenna_name) float64 ...
            const_rot   (time, antenna_name) float64 ...
            J           (time, antenna_name, frequency, polarization) ...
        Attributes:
            data_model:     RotationMeasureData
    """

    __slots__ = ()

    def __init__(self, data_vars=None, coords=None, attrs=None):
        super().__init__(data_vars=data_vars, coords=coords, attrs=attrs)

    @classmethod
    def constructor(
        cls,
        time: np.ndarray,
        antenna_name: np.ndarray,
        frequency: np.ndarray,
        resolution: np.ndarray,
        polarization: np.ndarray,
        lambda_sq: np.ndarray,
        rm_spec: np.ndarray | da.Array,
        rm_est: np.ndarray | da.Array,
        rm_peak: np.ndarray | da.Array,
        const_rot: np.ndarray | da.Array,
        J: np.ndarray | da.Array,
    ) -> "RotationMeasureData":
        """
        Assemble arrays into a RotationMeasureData dataset container.

        Parameters
        ----------
        time
            Array of time coordinates.
        antenna_name
            Array of antenna name coordinates.
        frequency
            Array of frequency coordinates.
        resolution
            Array of resolution coordinates.
        polarization
            Array of polarization coordinates.
        lambda_sq
            Array of squared wavelength values.
        rm_spec
            Rotation measure spectrum data array.
        rm_est
            Estimated rotation measure data array.
        rm_peak
            Peak rotation measure data array.
        const_rot
            Constant rotation data array.
        J
            Jones matrix data array.

        Returns
        -------
            The constructed rotation measure dataset.
        """
        data_vars = dict(
            lambda_sq=(["frequency"], lambda_sq),
            rm_spec=(["time", "antenna_name", "resolution"], rm_spec),
            rm_est=(["time", "antenna_name"], rm_est),
            rm_peak=(["time", "antenna_name"], rm_peak),
            const_rot=(["time", "antenna_name"], const_rot),
            J=(["time", "antenna_name", "frequency", "polarization"], J),
        )

        coords = dict(
            time=time,
            antenna_name=antenna_name,
            frequency=frequency,
            resolution=resolution,
            polarization=polarization,
        )

        attrs = {
            "data_model": "RotationMeasureData",
        }

        return cls(data_vars=data_vars, coords=coords, attrs=attrs)


def get_plot_params_for_station(
    dataset: RotationMeasureData,
    antenna: int | None,
    refant: int,
    time: int = 0,
) -> dict:
    """
    Extract plotting parameters natively retaining lazy Dask
    evaluation structures.

    Parameters
    ----------
    dataset
        Dataset containing results of model rotation calculation
    antenna
        Antenna (station) number to plot. If None, the station with the
        largest absolute RM relative to ``refant`` is chosen, over all
        selected solution intervals. The choice stays lazy.
    refant
        Reference antenna used during computation
    time
        Solution interval index to filter on

    Returns
    -------
        A dictionary with following keys and values data types:

        - stn: int, or a lazy dask scalar if ``antenna`` is None
        - rm_vals: np.ndarray
        - lambda_sq: dask.array
        - rm_spec: dask.array
        - rm_peak: dask.array
        - rm_est: dask.array
        - rm_est_refant: dask.array
        - J: dask.array
        - xlim: dask.array
    """
    ds = dataset.isel(time=time) if time is not None else dataset

    if antenna is None:
        # Station with the largest |RM| relative to the reference antenna,
        # picked lazily so nothing is computed before the plot runs.
        rm_rel = ds["rm_est"] - ds["rm_est"].isel(antenna_name=refant)
        rm_rel_abs = abs(rm_rel)
        if "time" in rm_rel_abs.dims:
            rm_rel_abs = rm_rel_abs.max("time")
        antenna = da.argmax(da.asarray(rm_rel_abs.data))

    def _station(name):
        variable = ds[name]
        return da.take(
            da.asarray(variable.data),
            antenna,
            axis=variable.get_axis_num("antenna_name"),
        )

    rm_spec = _station("rm_spec")
    rm_peak = _station("rm_peak")
    rm_est = _station("rm_est")
    rm_est_refant = ds["rm_est"].isel(antenna_name=refant).data
    J_val = pol_to_jones(_station("J"))

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
    frequency: np.ndarray, oversample: int, max_rm: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute wavelength squared and rotation measure arrays using NumPy.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency values in Hz, in ascending or descending order.
        Shape: (freq,)
    oversample : int
        Oversampling factor.
    max_rm : float, optional
        Upper limit on the absolute RM of the search grid, in rad/m^2. If
        None, the limit is set by the smallest channel spacing in
        wavelength squared, which gives very large grids for narrow
        channels.

    Returns
    -------
    lambda_sq : np.ndarray
        Wavelength squared values. Shape: (freq,)
    rm_vals : np.ndarray
        Rotation measure search grid values. Shape: (resolution,)
    """
    frequency = np.asarray(frequency)
    if frequency.size < 2:
        raise ValueError(
            "At least two frequency channels are needed to fit a rotation "
            f"measure, received {frequency.size}."
        )

    lambda_sq = (
        (const.c.value / frequency) ** 2  # pylint: disable=E1101
    ).astype(np.float32)

    # The smallest spacing sets the largest unambiguous RM. Taking the
    # absolute value makes this independent of the channel order.
    min_spacing = np.min(np.abs(np.diff(lambda_sq)))
    if min_spacing <= 0:
        raise ValueError(
            "Frequency channels must be distinct to fit a rotation measure."
        )

    rm_res = 1 / oversample / (np.max(lambda_sq) - np.min(lambda_sq))
    rm_max = 1 / min_spacing
    if max_rm is not None:
        rm_max = min(rm_max, max_rm)
    rm_max = np.ceil(rm_max / rm_res) * rm_res

    rm_vals = np.arange(
        -rm_max,
        rm_max,
        rm_res,
        dtype=np.float32,
    )
    return lambda_sq, rm_vals


def model_rotations(
    gaintable: GainCalibrationSetXds,
    peak_threshold: float = 0.5,
    refine_fit: bool = True,
    refant: int = 0,
    oversample: int = 5,
    max_rm: float | None = None,
) -> RotationMeasureData:
    """
    Fit a rotation measure for each station Jones matrix.

    For each station, the approach discussed in appendix B of de Gasperin et
    al. (2019) A&A 622, A5, is used to estimate the angle of rotation in each
    Jones matrix as a function of frequency. The angle is only defined
    modulo pi, so it is expressed as the phasor ``exp(2i * angle)`` and
    Fourier transformed with respect to wavelength squared. The peak of this
    RM spectrum is taken as the rotation measure for the station, and
    optionally refined by locating the peak between grid points, which is
    the least-squares fit of ``RM * lambda^2 + const`` to the angles.

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
        Reference antenna index
    oversample
        Oversampling value used in the rotation
        calculation. This determines the resolution of phasor.
        Note that setting this value to some higher
        integer may result in high memory usage.
    max_rm
        Upper limit on the absolute RM searched, in rad/m^2. If None, the
        limit follows from the channel spacing.

    Returns
    -------
        A dataset holding RM estimates and other data computed
    """
    gaintable = gaintable.chunk(time=1, antenna_name=1, frequency=-1)
    gaintable_refant = gaintable.isel(antenna_name=refant, drop=True)

    lambda_sq, rm_vals_coords = compute_rm_parameters(
        gaintable["frequency"].values, oversample, max_rm
    )

    n_time = gaintable.sizes["time"]
    n_ant = gaintable.sizes["antenna_name"]
    n_res = rm_vals_coords.size
    n_freq = gaintable.sizes["frequency"]
    n_pol = gaintable.sizes["polarization"]

    time_chunks = gaintable.chunks["time"]
    ant_chunks = gaintable.chunks["antenna_name"]

    rm_spec_da = da.empty(
        (n_time, n_ant, n_res),
        chunks=(time_chunks, ant_chunks, n_res),
        dtype=np.complex128,
    )
    rm_est_da = da.empty(
        (n_time, n_ant),
        chunks=(time_chunks, ant_chunks),
        dtype=np.float64,
    )
    rm_peak_da = da.empty(
        (n_time, n_ant),
        chunks=(time_chunks, ant_chunks),
        dtype=np.float64,
    )
    const_rot_da = da.empty(
        (n_time, n_ant),
        chunks=(time_chunks, ant_chunks),
        dtype=np.float64,
    )
    J_da = da.empty(
        (n_time, n_ant, n_freq, n_pol),
        chunks=(time_chunks, ant_chunks, n_freq, n_pol),
        dtype=gaintable["CALPARAM_GAIN"].dtype,
    )

    template = RotationMeasureData.constructor(
        time=gaintable.coords["time"].values,
        antenna_name=gaintable.coords["antenna_name"].values,
        frequency=gaintable.coords["frequency"].values,
        resolution=rm_vals_coords,
        polarization=gaintable.coords["polarization"].values,
        lambda_sq=lambda_sq,
        rm_spec=rm_spec_da,
        rm_est=rm_est_da,
        rm_peak=rm_peak_da,
        const_rot=const_rot_da,
        J=J_da,
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

    rm_vals = template_block.coords["resolution"].values
    lambda_sq = template_block["lambda_sq"].values

    # Drop 1-element time and antenna dimensions for core block extraction
    gaintable_squeezed = gaintable.squeeze(dim=["time", "antenna_name"])
    gaintable_refant_squeezed = gaintable_refant.squeeze(dim="time")

    # Extract raw NumPy arrays from the squeezed Xarray Datasets
    gain = pol_to_jones(gaintable_squeezed["CALPARAM_GAIN"].values)
    weight = pol_to_jones(gaintable_squeezed["CALPARAM_WEIGHT"].values)
    gain_refant = pol_to_jones(
        gaintable_refant_squeezed["CALPARAM_GAIN"].values
    )
    weight_refant = pol_to_jones(
        gaintable_refant_squeezed["CALPARAM_WEIGHT"].values
    )

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

    block_ds = template_block.assign(
        rm_spec=(
            ["time", "antenna_name", "resolution"],
            rm_spec[np.newaxis, np.newaxis, :],
        ),
        rm_est=(
            ["time", "antenna_name"],
            np.array([[rm_est]], dtype=np.float64),
        ),
        rm_peak=(
            ["time", "antenna_name"],
            np.array([[rm_peak]], dtype=np.float64),
        ),
        const_rot=(
            ["time", "antenna_name"],
            np.array([[const_rot]], dtype=np.float64),
        ),
        J=(
            ["time", "antenna_name", "frequency", "polarization"],
            jones_to_pol(J)[np.newaxis, np.newaxis, ...],
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
    using pure NumPy arrays. This can be treated as a ufunc, and can be
    used to broadcast computations across stations and time.

    Parameters
    ----------
    gain
        Calibrated gain matrix array for a single station block.
        Shape: (freq, rec1, rec2)
    weight
        Weight matrix array for a single station block.
        Shape: (freq, rec1, rec2)
    gain_refant
        Calibrated gain matrix array for the reference station.
        Shape: (freq, rec1, rec2)
    weight_refant
        Weight matrix array for the reference station.
        Shape: (freq, rec1, rec2)
    rm_vals
        Pre-computed rotation measure search grid values. Shape: (resolution,).
    lambda_sq
        Pre-computed wavelength squared values. Shape: (freq,).
    frequency
        Frequency values in Hz. Only utilized if rm_vals/lambda_sq
        must be computed. Shape: (freq,).
    oversample
        Oversampling value used if parameters must be generated on the fly.
        Only utilized if rm_vals/lambda_sq must be computed.
    peak_threshold
        Height of peak in the RM spectrum required for a rotation detection.
    refine_fit
        Whether or not to refine the RM spectrum peak locations with a
        nonlinear optimisation.

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

    # Channels with non-finite gains carry no information. Zero them so
    # they cannot spread NaNs, and leave them out of the mask below.
    finite = np.all(np.isfinite(J), axis=(1, 2))  # Shape: (freq,)
    J[~finite] = 0

    norms = np.linalg.norm(
        J, axis=(1, 2), keepdims=True
    )  # Shape: (freq,rec1,rec2)
    mask = get_stn_masks(weight, weight_refant) & finite  # Shape: (freq,)
    mask = mask & (norms[:, 0, 0] > 0)  # Shape: (freq,)

    J = update_jones_with_masks(J, mask, norms)  # Shape: (freq,rec1,rec2)
    phi_raw = calculate_phi_raw(J)  # Shape: (freq,)

    rm_spec = get_rm_spec(
        phi_raw, mask, rm_vals, lambda_sq
    )  # Shape: (resolution,)
    abs_rm_spec = np.abs(rm_spec)

    # The reference station has no rotation relative to itself.
    is_refant = np.array_equal(gain, gain_refant)
    detected = (
        bool(np.any(mask))
        and not is_refant
        and np.max(abs_rm_spec) > peak_threshold
    )

    rm_est = float(rm_vals[np.argmax(abs_rm_spec)]) if detected else 0.0

    rm_peak = rm_est
    const_rot = 0.0

    # Without a detection there is no peak to refine: keep RM = 0 rather
    # than fitting noise.
    if refine_fit and detected:
        rm_step = float(rm_vals[1] - rm_vals[0]) if rm_vals.size > 1 else 1.0
        rm_est, const_rot = fit_curve(
            lambda_sq, phi_raw, rm_peak, mask=mask, rm_step=rm_step
        )

    return J, rm_spec, rm_est, rm_peak, const_rot


def calculate_phi_raw(jones: np.ndarray) -> np.ndarray[float]:
    """
    Parameters
    ----------
    jones
        Jones array for given antenna (freq, rec1, rec2)

    Returns
    -------
        Rotation angle per channel in radians, wrapped to
        ``(-pi/2, pi/2]``. Shape: (freq,). float

    Notes
    -----
    In the circular basis a rotation by ``phi`` gives the two hands phases
    of ``-phi`` and ``+phi``, so half their phase difference is ``phi``.
    The difference is taken as the angle of one product per channel. This
    cancels any phase common to both hands exactly, without unwrapping
    along frequency, so noisy or flagged channels cannot affect their
    neighbours. The angle is therefore only known modulo pi, which is all
    a rotation of an unpolarised model can determine.
    """
    co_sum = jones[:, 0, 0] + jones[:, 1, 1]
    cross_diff = 1j * (jones[:, 0, 1] - jones[:, 1, 0])
    return 0.5 * np.angle((co_sum + cross_diff) * np.conj(co_sum - cross_diff))


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
        Rotation angle per channel, defined modulo pi. Shape: ``(freq,)``.
        float
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
        Array of RM spec. Shape: ``(resolution)``. complex. All zeros if
        every channel is masked.

    Notes
    -----
    The angle is only defined modulo pi, so the transform uses the phasor
    ``exp(2i * phi)`` against ``exp(-2i * lambda^2 * RM)``.
    """
    resolution = rm_vals.shape[0]

    phi_exp = np.exp(2j * phi_raw)
    masked_phi = np.where(mask, phi_exp, 0j)
    counts = mask.sum(dtype=np.float32)

    if counts == 0:
        return np.zeros((resolution,), dtype=np.complex128)

    result = np.empty((resolution,), dtype=np.complex128)

    # Process the resolution axis in chunks
    for i in range(0, resolution, chunk_size):
        end = min(i + chunk_size, resolution)
        rm_chunk = rm_vals[i:end]

        # Intermediate array shape: (nchannels, chunk_size)
        # Instead of allocating (nchannels, resolution) all at once
        phasor_chunk = np.exp(
            -2j * lambda_sq[:, np.newaxis] * rm_chunk[np.newaxis, :]
        )
        result[i:end] = masked_phi @ phasor_chunk

    return result / counts


def fit_curve(
    lambda_sq: np.ndarray,
    phi_raw: np.ndarray,
    rm_est: float,
    mask: np.ndarray | None = None,
    rm_step: float = 1.0,
) -> tuple[float, float]:
    """
    Refine a rotation measure estimate between RM grid points.

    Fits ``phi = rm * lambda^2 + const`` to the rotation angles in the
    least-squares sense on the unit circle. Because the angles are only
    known modulo pi, the fit is made to ``exp(2i * phi)``. For any ``rm``
    the best constant has a closed form, and the remaining cost is
    minimised where ``|sum(exp(2i * phi) * exp(-2i * lambda^2 * rm))|`` is
    largest, i.e. at the peak of the continuous RM spectrum. That peak is
    located with a bounded one-dimensional search, which cannot diverge.

    Parameters
    ----------
    lambda_sq
        Wavelength square. (freq,) float
    phi_raw
        Rotation angle per channel, modulo pi. (freq,) float
    rm_est
        Initial rm estimate for current station, usually the RM grid peak.
        Shape: ()
    mask
        Channels to use in the fit. (freq,) bool. If None, all channels
        with a finite angle are used.
    rm_step
        Half-width of the search interval around ``rm_est``, usually the
        RM grid spacing.

    Returns
    -------
        (rm_est, const_rot)
        New rotation measure estimate and the constant rotation angle,
        which is defined modulo pi and returned in ``(-pi/2, pi/2]``.
    """
    phi_raw = np.asarray(phi_raw, dtype=float)
    lambda_sq = np.asarray(lambda_sq, dtype=float)

    use = np.isfinite(phi_raw)
    if mask is not None:
        use = use & np.asarray(mask, dtype=bool)

    if not np.any(use):
        return float(rm_est), 0.0

    phasor = np.exp(2j * phi_raw[use])
    lambda_sq = lambda_sq[use]

    def spectrum(rm):
        return np.sum(phasor * np.exp(-2j * lambda_sq * rm))

    result = minimize_scalar(
        lambda rm: -np.abs(spectrum(rm)),
        bounds=(rm_est - rm_step, rm_est + rm_step),
        method="bounded",
        options={"xatol": 1e-10 * max(1.0, abs(rm_step))},
    )
    rm_fit = float(result.x)
    const_rot = 0.5 * float(np.angle(spectrum(rm_fit)))
    return rm_fit, const_rot


# ITU-R P.531 eq. 2, as used by OSKAR's TEC-screen Faraday kernel:
# angle [rad] = 236 * B_parallel [T] * TEC [TECU] / f [GHz]^2.
# With angle = RM * lambda^2, RM [rad/m^2] = 236e18 / c^2 * B * TEC.
FARADAY_RM_PER_TESLA_TECU = (
    2.36e2 * 1e18 / const.c.value**2  # pylint: disable=E1101
)


def _station_geodetic(
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert station ITRF positions to WGS84 geodetic coordinates.

    Parameters
    ----------
    xyz
        Station ITRF/ECEF positions in metres, shape ``(antenna, 3)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Longitude and latitude in radians, and height above the ellipsoid
        in metres, each of shape ``(antenna,)``.
    """
    xyz = np.asarray(xyz, dtype=float)
    location = EarthLocation.from_geocentric(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], unit="m"
    )
    lon, lat, height = location.to_geodetic("WGS84")
    return (
        np.asarray(lon.rad),
        np.asarray(lat.rad),
        np.asarray(height.to_value("m")),
    )


def _station_magnetic_field_enu(
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    height_m: np.ndarray,
    time_mjd: float,
) -> np.ndarray:
    """
    Evaluate the IGRF-14 geomagnetic field at each station.

    This matches OSKAR, which evaluates IGRF once per station at its
    geodetic ground position for the observation date.

    Parameters
    ----------
    lon_rad, lat_rad
        Geodetic longitude and latitude in radians, shape ``(antenna,)``.
    height_m
        Height above the WGS84 ellipsoid in metres, shape ``(antenna,)``.
    time_mjd
        UTC date of the observation as MJD (days).

    Returns
    -------
    np.ndarray
        Field (east, north, up) in nT, shape ``(antenna, 3)``.
    """
    date = Time(time_mjd, format="mjd", scale="utc").to_datetime()
    # ppigrf returns arrays of shape (n_dates, antenna); there is one date.
    b_east, b_north, b_up = ppigrf.igrf(
        np.degrees(lon_rad),
        np.degrees(lat_rad),
        np.asarray(height_m) / 1000.0,
        date,
    )
    return np.stack(
        [np.ravel(b_east), np.ravel(b_north), np.ravel(b_up)], axis=-1
    )


def _phasecentre_enu(
    phasecentre,
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    times_mjd: np.ndarray,
) -> np.ndarray:
    """
    Unit vector towards the phase centre in each station's local ENU frame.

    The hour angle uses Greenwich apparent sidereal time plus the station
    longitude, consistent with the TEC-screen geometry.

    Parameters
    ----------
    phasecentre
        Phase centre as a ``SkyCoord``.
    lon_rad, lat_rad
        Station geodetic longitude and latitude in radians,
        shape ``(antenna,)``.
    times_mjd
        UTC times in MJD (days), shape ``(time,)``.

    Returns
    -------
    np.ndarray
        Direction cosines (east, north, up), shape ``(time, antenna, 3)``.
    """
    gast = (
        Time(times_mjd, format="mjd", scale="utc")
        .sidereal_time("apparent", "greenwich")
        .rad
    )
    ra = phasecentre.ra.rad
    dec = phasecentre.dec.rad

    # Local hour angle, shape (time, antenna).
    ha = gast[:, np.newaxis] + lon_rad[np.newaxis, :] - ra
    sin_lat = np.sin(lat_rad)[np.newaxis, :]
    cos_lat = np.cos(lat_rad)[np.newaxis, :]

    east = -np.cos(dec) * np.sin(ha)
    north = cos_lat * np.sin(dec) - sin_lat * np.cos(dec) * np.cos(ha)
    up = sin_lat * np.sin(dec) + cos_lat * np.cos(dec) * np.cos(ha)
    return np.stack([east, north, up], axis=-1)


def rm_from_external_tec(
    vis: Visibility,
    external_tec_fits: str,
    solution_times: np.ndarray,
) -> np.ndarray:
    """
    Compute station rotation measures from an external OSKAR TEC screen.

    This reproduces the Faraday rotation OSKAR applies with a TEC screen
    (``oskar_evaluate_tec_screen.h``, ITU-R P.531 eq. 2):

    - TEC is sampled at each station's phase-centre pierce point, as in
      :func:`sample_external_tec`.
    - The IGRF-14 field at the station is projected on the line of sight.
      OSKAR takes the propagation direction (source to observer), so
      ``B_parallel = -(s . B)`` with ``s`` the direction to the source.
    - OSKAR's rotation angle is ``236e18 / c^2 * B_parallel[T] *
      TEC[TECU] * lambda^2``, applied in its station beam basis.

    The returned RM is the negative of OSKAR's coefficient. OSKAR applies
    the rotation in its station beam basis, while the pipeline applies
    :func:`generate_rotation_matrices` in the sky frame of the beam model
    (E = Beam @ R), which rotates in the opposite sense. With the sign
    flipped, the re-solved bandpass on OSKAR data simulated with
    aperture-array beams keeps the same residual rotation as with the
    gain-based estimate (``model_rotations``).

    This assumes a beam model is used in prediction. Without one, R acts
    in the feed frame, where OSKAR's own sign applies; for isotropic-beam
    simulations predicted without beams, negate the result.

    The phase centre is used as the source direction for every station,
    which is exact for a calibrator at the phase centre.

    Parameters
    ----------
    vis
        Visibility dataset with the configuration, phase centre, timing and
        ``ARRAY_CENTER`` metadata required by :func:`sample_external_tec`.
    external_tec_fits
        Path to the external FITS TEC-screen cube used by OSKAR.
    solution_times
        UTC MJD times in seconds, shape ``(time,)``.

    Returns
    -------
    np.ndarray
        Absolute (not reference-relative) rotation measures in rad/m^2,
        shape ``(time, antenna)``, antennas in configuration order.
    """
    solution_times = np.atleast_1d(np.asarray(solution_times, dtype=float))
    times_mjd = solution_times / 86400.0

    tec = sample_external_tec(vis, external_tec_fits, solution_times)

    lon, lat, height = _station_geodetic(vis.configuration.xyz.data)

    # OSKAR evaluates the field once, at the start of the observation.
    field_enu_nt = _station_magnetic_field_enu(lon, lat, height, times_mjd[0])
    source_enu = _phasecentre_enu(vis.phasecentre, lon, lat, times_mjd)

    # Field along the propagation direction, nT -> T.
    b_parallel_t = (
        -np.sum(source_enu * field_enu_nt[np.newaxis, :, :], axis=-1) * 1e-9
    )

    # Negated: OSKAR's rotation is in the beam basis, the pipeline's R is
    # in the beam model's sky frame, which has the opposite handedness.
    rm = -FARADAY_RM_PER_TESLA_TECU * b_parallel_t * tec

    logger.info(
        "Rotation measures from external TEC: min %.4f, max %.4f rad/m^2 "
        "over %d times and %d antennas.",
        float(np.min(rm)),
        float(np.max(rm)),
        rm.shape[0],
        rm.shape[1],
    )
    return rm
