import logging

import numpy as np
from astropy import constants as const
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.sky_model import SkyComponent

logger = logging.getLogger(__name__)


def generate_rotation_matrices(
    rm: np.ndarray,
    frequency: np.ndarray,
    output_dtype: type = np.complex64,
) -> np.ndarray:
    """Generate station rotation matrix from RM values.

    :param rm: 1D array of rotation measure values [nstation].
    :param frequency: 1D array of frequency values [nfrequency].
    :param output_dtype: output dtype of rotation matrix

    :return: 4D array of rotation matrix: [nstation, nfrequency, 2, 2].
    """
    lambda_sq = np.power(
        (const.c.value / frequency), 2  # pylint: disable=E1101
    )

    phi = rm[..., np.newaxis] * lambda_sq

    cos_val = np.cos(phi)
    sin_val = np.sin(phi)

    eye = np.array([[1, 0], [0, 1]])
    A = np.array([[0, -1], [1, 0]])

    rot_array = (
        cos_val[:, :, np.newaxis, np.newaxis] * eye
        + sin_val[:, :, np.newaxis, np.newaxis] * A
    )

    return rot_array.astype(output_dtype)


def gaussian_tapers(
    u: np.ndarray,
    v: np.ndarray,
    params: dict[float],
) -> np.ndarray:
    """Calculated visibility amplitude tapers for Gaussian components.

    Note: this needs to be tested. Generate, image and fit a model component?
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
    uvw: (time, baselineid, spatial)
    skycomponent.frequency: (frequency,)
    skycomponent.flux: (frequency, polarisation)

    returns: (time, baselineid, frequency, polarisation)
    """

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
