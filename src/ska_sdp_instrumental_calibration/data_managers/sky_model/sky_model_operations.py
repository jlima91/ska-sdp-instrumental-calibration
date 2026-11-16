"""
Generic operations on the ``ska_sdp_datamodels.sky_model``
dataclasses (e.g. ``SkyComponent``).

These functions do not depend on anything specific to this repository, and
are intended to be moved to ska-sdp-func-python
(``ska_sdp_func_python.sky_component``).

The DFT functions replace the removed
``ska_sdp_func_python.imaging.dft_skycomponent``, which operated on the old
xarray-based ``SkyComponent``. They use the same parameter names and units
as ``ska_sdp_datamodels.sky_model.SkyComponent``.
"""

__all__ = [
    "deconvolve_gaussian",
    "dft_skycomponent",
    "direction_cosines",
    "gaussian_shape",
    "gaussian_tapers",
    "is_above_horizon",
    "skycomponent_flux",
]

from typing import Optional

import numpy as np
from astropy import constants as const
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.science_data_model.polarisation_functions import (
    convert_pol_frame,
)
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.sky_model.sky_model_operations import (
    calculate_flux_at_freq,
)


def direction_cosines(
    ra_deg: float, dec_deg: float, phase_centre: SkyCoord
) -> tuple[float, float, float]:
    """
    Calculate direction cosines (l, m, n) of a position relative to the
    phase centre.

    Parameters
    ----------
    ra_deg : float
        Right ascension of the position. Units: [deg]
    dec_deg : float
        Declination of the position. Units: [deg]
    phase_centre : SkyCoord
        Phase centre of the observation.

    Returns
    -------
    tuple[float, float, float]
        Direction cosines (l, m, n). Dimensionless.
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    ra0 = phase_centre.ra.radian
    cdec0 = np.cos(phase_centre.dec.radian)
    sdec0 = np.sin(phase_centre.dec.radian)

    cdec = np.cos(dec)
    sdec = np.sin(dec)
    cdra = np.cos(ra - ra0)

    l_comp = cdec * np.sin(ra - ra0)
    m_comp = sdec * cdec0 - cdec * sdec0 * cdra
    n_comp = sdec * sdec0 + cdec * cdec0 * cdra

    return l_comp, m_comp, n_comp


def is_above_horizon(
    skycomponent: SkyComponent, obstime: Time, location: EarthLocation
) -> bool:
    """
    Check whether a sky component is above the horizon.

    Parameters
    ----------
    skycomponent : SkyComponent
        Sky component.
    obstime : Time
        Time of the observation.
    location : EarthLocation
        Location of the observer (e.g. the array).

    Returns
    -------
    bool
        True if the altitude of the component is at least zero.
    """
    direction = SkyCoord(
        ra=skycomponent.ra_deg, dec=skycomponent.dec_deg, unit="deg"
    )
    altaz = direction.transform_to(AltAz(obstime=obstime, location=location))

    return bool(altaz.alt.degree >= 0)


def gaussian_tapers(
    u: np.ndarray,
    v: np.ndarray,
    a_arcsec: float,
    b_arcsec: float,
    pa_deg: float,
) -> np.ndarray:
    """
    Calculate visibility amplitude tapers for a Gaussian component.

    This is a copy of ``ska_sdp_func_python.imaging.dft.gaussian_tapers``,
    modified to accept the shape parameters with the names and units of
    ``ska_sdp_datamodels.sky_model.SkyComponent``. It should be
    replaced once an equivalent function is available in
    ska-sdp-func-python.

    The taper is the Fourier transform of an elliptical Gaussian,
    normalised to unity at zero baseline length. The component flux is
    therefore treated as the integrated flux of the Gaussian.

    The shape parameters follow the GSM schema conventions:

    * ``a_arcsec`` and ``b_arcsec`` are the literal semi-major and
      semi-minor axes, i.e. half of the full width at half maximum (FWHM)
      along each axis.
    * ``pa_deg`` is the position angle of the major axis, measured from
      North through East (anticlockwise on the sky from North).

    Note that ``ska_sdp_func_python.imaging.dft.gaussian_tapers`` instead
    takes FWHM axes and measures the position angle from the u-axis
    towards the v-axis.

    Parameters
    ----------
    u : np.ndarray
        The u-coordinates of the baselines. Units: [wavelengths]
    v : np.ndarray
        The v-coordinates of the baselines. Must be broadcastable with
        ``u``. Units: [wavelengths]
    a_arcsec : float
        Semi-major axis (half of the FWHM) of the Gaussian.
        Units: [arcsec]
    b_arcsec : float
        Semi-minor axis (half of the FWHM) of the Gaussian.
        Units: [arcsec]
    pa_deg : float
        Position angle of the major axis, from North through East.
        Units: [deg]

    Returns
    -------
    np.ndarray
        Taper values in the range (0, 1], with the broadcast shape of
        ``u`` and ``v``. Dimensionless.
    """
    # exp(-a*x^2) transforms to exp(-pi^2*u^2/a)
    # a = 4log(2)/FWHM^2 so scaling = pi^2 * FWHM^2 / (4log(2))
    scale = -(np.pi * np.pi) / (4 * np.log(2.0))

    # Convert semi-axes to FWHM in radians
    bmaj = 2 * a_arcsec / 3600.0 * np.pi / 180
    bmin = 2 * b_arcsec / 3600.0 * np.pi / 180
    bpa = pa_deg * np.pi / 180

    # Rotate baselines to the major/minor axes. u points East and v points
    # North, so the major axis at position angle bpa (North through East)
    # lies along (sin(bpa), cos(bpa)) in the (u, v) plane.
    up = np.sin(bpa) * u + np.cos(bpa) * v
    vp = -np.cos(bpa) * u + np.sin(bpa) * v

    return np.exp((bmaj * bmaj * up * up + bmin * bmin * vp * vp) * scale)


def _value_or_none(value: Optional[float]) -> Optional[float]:
    """
    Return None for missing values. Missing float values are None when the
    SkyComponent is created directly, and NaN when it is read from a CSV
    file using ``read_skymodel_from_lsm_csv``.
    """
    if value is None or np.isnan(value):
        return None
    return value


def gaussian_shape(
    skycomponent: SkyComponent,
) -> Optional[tuple[float, float, float]]:
    """
    Get the Gaussian shape parameters of a sky component.

    The component is treated as a point source if either axis is missing
    (None or NaN), or if both axes are zero. A missing position angle is
    treated as zero.

    Parameters
    ----------
    skycomponent : SkyComponent
        Sky component.

    Returns
    -------
    Optional[tuple[float, float, float]]
        ``(a_arcsec, b_arcsec, pa_deg)`` for a Gaussian component, or None
        for a point source. Units: ([arcsec], [arcsec], [deg])
    """
    a_arcsec = _value_or_none(skycomponent.a_arcsec)
    b_arcsec = _value_or_none(skycomponent.b_arcsec)
    if (
        a_arcsec is None
        or b_arcsec is None
        or (a_arcsec == 0 and b_arcsec == 0)
    ):
        return None

    pa_deg = _value_or_none(skycomponent.pa_deg) or 0.0

    return a_arcsec, b_arcsec, pa_deg


def skycomponent_flux(
    skycomponent: SkyComponent,
    frequency: np.ndarray,
    polarisation_frame: PolarisationFrame,
) -> np.ndarray:
    """
    Calculate the flux of a sky component at the given frequencies, in the
    given polarisation frame.

    The sky component only provides Stokes I flux, which is converted to the
    requested polarisation frame. For the linear frame this gives
    XX = YY = I and XY = YX = 0.

    Parameters
    ----------
    skycomponent : SkyComponent
        Sky component.
    frequency : np.ndarray
        Frequencies of shape (n_freqs,). Units: [Hz]
    polarisation_frame : PolarisationFrame
        Polarisation frame of the output flux.

    Returns
    -------
    np.ndarray
        Flux of shape (n_freqs, n_pols). Units: [Jy]
    """
    flux = calculate_flux_at_freq(skycomponent, frequency).reshape(-1, 1)

    stokes_i = PolarisationFrame("stokesI")
    if polarisation_frame == stokes_i:
        return flux

    return convert_pol_frame(flux, stokes_i, polarisation_frame)


def dft_skycomponent(
    uvw: np.ndarray,
    skycomponent: SkyComponent,
    phase_centre: SkyCoord,
    frequency: np.ndarray,
    polarisation_frame: PolarisationFrame = PolarisationFrame("linear"),
) -> np.ndarray:
    """
    Calculate the Direct Fourier Transform (DFT) for a single sky component.

    This function computes the visibility contribution of a sky component by
    calculating the phase delay associated with its position relative to the
    phase centre. It accounts for wide-field effects (w-term) and applies
    Gaussian tapering if the component has non-zero shape parameters.

    Parameters
    ----------
    uvw : np.ndarray
        The UVW coordinates of the baselines, of shape
        (n_times, n_baselines, 3). Units: [m]
    skycomponent : SkyComponent
        The sky component to transform.
    phase_centre : SkyCoord
        The phase centre of the observation, used as the reference point for
        calculating direction cosines (l, m, n).
    frequency : np.ndarray
        Frequencies at which to calculate visibilities, of shape (n_freqs,).
        Units: [Hz]
    polarisation_frame : PolarisationFrame, default: linear
        Polarisation frame of the output visibilities.

    Returns
    -------
    np.ndarray
        The calculated visibilities for the component, of shape
        (n_times, n_baselines, n_freqs, n_pols). Units: [Jy]
    """
    frequency = np.asarray(frequency)

    scaled_uvw = np.einsum(
        "tbs,f->tbfs",
        uvw,
        frequency / const.c.value,  # pylint: disable=no-member
    )
    scaled_u = scaled_uvw[..., 0]
    scaled_v = scaled_uvw[..., 1]
    scaled_w = scaled_uvw[..., 2]

    l_comp, m_comp, n_comp = direction_cosines(
        skycomponent.ra_deg, skycomponent.dec_deg, phase_centre
    )

    comp_data = np.exp(
        -2j
        * np.pi
        * (scaled_u * l_comp + scaled_v * m_comp + scaled_w * (n_comp - 1))
    )

    shape = gaussian_shape(skycomponent)
    if shape is not None:
        comp_data = comp_data * gaussian_tapers(scaled_u, scaled_v, *shape)

    return np.einsum(
        "tbf,fp->tbfp",
        comp_data,
        skycomponent_flux(skycomponent, frequency, polarisation_frame),
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals
def deconvolve_gaussian(
    a_arcsec: float,
    b_arcsec: float,
    pa_deg: float,
    beam_a_arcsec: float,
    beam_b_arcsec: float,
    beam_pa_deg: float,
) -> tuple[float, float, float]:
    """
    Deconvolve a synthesised beam from fitted Gaussian shape parameters.

    This follows the approach of the analysisutilities function
    deconvolveGaussian in the askap-analysis repository, written by Matthew
    Whiting. This is based on the approach described in Wild (1970),
    AuJPh 23, 113.

    If a minor axis is larger than its major axis, the corresponding
    Gaussian is treated as circular, with both axes equal to the major
    axis. Deconvolved axes which would be imaginary (i.e. the fitted
    Gaussian is smaller than the beam along that axis) are set to zero.

    The axes and position angles of the fitted Gaussian and of the beam
    must follow the same conventions (e.g. all FWHM, with position angles
    measured from the same reference direction). The deconvolution only
    depends on the squares of the axes, so the deconvolved axes are returned
    in the same convention as the input axes: FWHM inputs give FWHM outputs,
    semi-axis inputs give semi-axis outputs.

    The returned values are intended to populate ``a_arcsec``, ``b_arcsec``
    and ``pa_deg`` of ``ska_sdp_datamodels.sky_model.SkyComponent``,
    which the GSM schema defines as the deconvolved Gaussian semi-major
    axis, semi-minor axis (literal semi-axes, i.e. half of the FWHM) and
    position angle (from North through East). Inputs in FWHM, such as the
    major / minor axis columns of the GLEAM catalogue, must therefore be
    halved, either before or after the deconvolution.

    Parameters
    ----------
    a_arcsec : float
        Fitted major axis of the source. Units: [arcsec]
    b_arcsec : float
        Fitted minor axis of the source. Units: [arcsec]
    pa_deg : float
        Fitted position angle of the source. Units: [deg]
    beam_a_arcsec : float
        Major axis of the beam at the source location. Units: [arcsec]
    beam_b_arcsec : float
        Minor axis of the beam at the source location. Units: [arcsec]
    beam_pa_deg : float
        Position angle of the beam at the source location. Units: [deg]

    Returns
    -------
    tuple[float, float, float]
        Deconvolved ``(a_arcsec, b_arcsec, pa_deg)`` of the source, i.e. the
        deconvolved Gaussian major axis, minor axis and position angle, in
        the same conventions as the inputs. Units: ([arcsec], [arcsec], [deg])
    """
    # fitted data on source
    fmajsq = a_arcsec * a_arcsec
    fminsq = b_arcsec * b_arcsec if b_arcsec < a_arcsec else fmajsq
    fdiff = fmajsq - fminsq
    fphi = 2.0 * pa_deg * np.pi / 180.0

    # beam data at source location
    bmajsq = beam_a_arcsec * beam_a_arcsec
    bminsq = (
        beam_b_arcsec * beam_b_arcsec
        if beam_b_arcsec < beam_a_arcsec
        else bmajsq
    )
    bdiff = bmajsq - bminsq
    bphi = 2.0 * beam_pa_deg * np.pi / 180.0

    # source data after deconvolution
    if fdiff < 1e-6:
        # Circular Gaussian case
        smajsq = fmajsq - bminsq
        sminsq = fmajsq - bmajsq
        psmaj = np.pi / 2.0 + beam_pa_deg * np.pi / 180.0

    else:
        # General case
        sinsphi = fdiff * np.sin(fphi) - bdiff * np.sin(bphi)
        cossphi = fdiff * np.cos(fphi) - bdiff * np.cos(bphi)
        sdiff = np.sqrt(
            fdiff * fdiff
            + bdiff * bdiff
            - 2.0 * fdiff * bdiff * np.cos(fphi - bphi)
        )
        smajsq = 0.5 * (fmajsq + fminsq - bmajsq - bminsq + sdiff)
        sminsq = 0.5 * (fmajsq + fminsq - bmajsq - bminsq - sdiff)
        psmaj = 0 if cossphi == 0 else np.arctan2(sinsphi, cossphi) / 2.0

    smaj = 0 if smajsq <= 0 else np.sqrt(smajsq)
    smin = 0 if sminsq <= 0 else np.sqrt(sminsq)

    return max(smaj, smin, 0), max(min(smaj, smin), 0), psmaj * 180 / np.pi
