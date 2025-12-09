# pylint: disable=W1401

import logging

import numpy as np
from astropy import constants as const
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.sky_model import SkyComponent

logger = logging.getLogger(__name__)


def gaussian_tapers(
    u: np.ndarray,
    v: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """
    Calculate visibility amplitude tapers for Gaussian components.

    This function computes the Gaussian tapering factor in the UV plane
    corresponding to a source with specific major/minor axes and position
    angle. It rotates the input UV coordinates to align with the Gaussian's
    orientation and applies the appropriate scaling based on the Full Width
    at Half Maximum (FWHM).

    Parameters
    ----------
    u : numpy.ndarray
        The u-coordinates of the baselines (typically in wavelengths).
    v : numpy.ndarray
        The v-coordinates of the baselines (typically in wavelengths).
    params : dict[str, float]
        A dictionary containing the Gaussian component parameters:

        * `bmaj`: The major axis FWHM in degrees.
        * `bmin`: The minor axis FWHM in degrees.
        * `bpa`: The position angle in degrees.

    Returns
    -------
    numpy.ndarray
        The calculated amplitude taper values. The shape matches the input
        `u` and `v` arrays.

    Notes
    -----
    The scaling factor is derived from the properties of the Fourier transform
    of a Gaussian function.

    **Development Note:** This implementation requires further validation.
    The recommended testing strategy is to generate a model component, image it
    and fit the resulting source to verify the taper accuracy.
    """
    # exp(-a*x^2) transforms to exp(-pi^2*u^2/a)
    # a = 4log(2)/FWHM^2 so scaling = pi^2 * FWHM^2 / (4log(2))
    scale = -(np.pi * np.pi) / (4 * np.log(2.0))
    # Rotate baselines to the major/minor axes:
    bpa = params["bpa"] * np.pi / 180
    bmaj = params["bmaj"] * np.pi / 180
    bmin = params["bmin"] * np.pi / 180

    up = np.cos(bpa) * u + np.sin(bpa) * v
    vp = -np.sin(bpa) * u + np.cos(bpa) * v

    return np.exp((bmaj * bmaj * up * up + bmin * bmin * vp * vp) * scale)


def dft_skycomponent(
    uvw: np.ndarray,
    skycomponent: SkyComponent,
    phase_centre: SkyCoord,
) -> np.ndarray:
    """
    Calculate the Direct Fourier Transform (DFT) for a single sky component.

    This function computes the visibility contribution of a sky component by
    calculating the phase delay associated with its position relative to the
    phase centre. It accounts for wide-field effects (w-term) and applies
    Gaussian tapering if the component shape is defined as such.

    Parameters
    ----------
    uvw : numpy.ndarray
        The UVW coordinates of the baselines in metres.
        Expected shape: (n_times, n_baselines, 3).
    skycomponent : SkyComponent
        The sky component object to transform. It must contain the following
        attributes:

        * ``frequency``: Array of shape (n_freqs,).
        * ``flux``: Array of shape (n_freqs, n_pols).
        * ``direction``: SkyCoord object representing the source position.
        * ``shape``: String indicating shape (e.g., "GAUSSIAN", "POINT").
        * ``params``: Dictionary of shape parameters (if Gaussian).
    phase_centre : SkyCoord
        The phase centre of the observation, used as the reference point for
        calculating direction cosines ($l, m, n$).

    Returns
    -------
    numpy.ndarray
        The calculated visibilities for the component.
        Shape: (n_times, n_baselines, n_freqs, n_pols).

    Notes
    -----
    The phase calculation uses the full $w$-term correction:

    .. math:: V = I \cdot \exp(-2\pi i (ul + vm + w(n-1)))

    """  # noqa: W605

    scaled_uvw = np.einsum(
        "tbs,f->tbfs",
        uvw,
        skycomponent.frequency / const.c.value,  # pylint: disable=no-member
    )
    scaled_u = scaled_uvw[..., 0]
    scaled_v = scaled_uvw[..., 1]
    scaled_w = scaled_uvw[..., 2]

    # Get coordaintes of phase centre
    ra0 = phase_centre.ra.radian
    cdec0 = np.cos(phase_centre.dec.radian)
    sdec0 = np.sin(phase_centre.dec.radian)

    cdec = np.cos(skycomponent.direction.dec.radian)
    sdec = np.sin(skycomponent.direction.dec.radian)
    cdra = np.cos(skycomponent.direction.ra.radian - ra0)
    l_comp = cdec * np.sin(skycomponent.direction.ra.radian - ra0)
    m_comp = sdec * cdec0 - cdec * sdec0 * cdra
    n_comp = sdec * sdec0 + cdec * cdec0 * cdra

    comp_data = np.exp(
        -2j
        * np.pi
        * (scaled_u * l_comp + scaled_v * m_comp + scaled_w * (n_comp - 1))
    )

    if skycomponent.shape == "GAUSSIAN":
        comp_data = comp_data * gaussian_tapers(
            scaled_u, scaled_v, skycomponent.params
        )

    return np.einsum(
        "tbf,fp->tbfp",
        comp_data,
        skycomponent.flux,
    )
