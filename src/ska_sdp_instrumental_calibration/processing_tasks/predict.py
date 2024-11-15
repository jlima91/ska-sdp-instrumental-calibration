"""Module for predicting model visibilities.
"""

__all__ = [
    "predict_from_components",
]

import importlib
import os

import numpy as np
import xarray as xr
from astropy import constants as const
from astropy.coordinates import AltAz
from astropy.time import Time
from numpy import typing
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.visibility.vis_model import Visibility
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.beams import (
    GenericBeams,
)

logger = setup_logger(__name__)


def gaussian_tapers(
    uvw: typing.NDArray[np.float_],
    params: dict[float],
) -> typing.NDArray[np.float_]:
    """Calculated visibility amplitude tapers for Gaussian components.

    Note that this need to be tested. Generate and image a model component...
    And compare with ska-sdp-func? No, seems to ignore.

    :param uvw: baseline coordinates ([time, baseline, frequency, dir]).
    :param params: dictionary of shape params {bmaj, bmin, bpa} in degrees.
    :return: visibility tapers ([time, baseline, frequency]).
    """
    # exp(-a*x^2) transforms to exp(-pi^2*u^2/a)
    # a = 4log(2)/FWHM^2 so scaling = pi^2 * FWHM^2 / (4log(2))
    scale = -(np.pi * np.pi) / (4 * np.log(2.0))
    # Rotate baselines to the major/minor axes:
    bpa = params["bpa"] * np.pi / 180
    bmaj = params["bmaj"] * np.pi / 180
    bmin = params["bmin"] * np.pi / 180
    up = np.cos(bpa) * uvw[..., 0] + np.sin(bpa) * uvw[..., 1]
    vp = -np.sin(bpa) * uvw[..., 0] + np.cos(bpa) * uvw[..., 1]
    return np.exp((bmaj * bmaj * up * up + bmin * bmin * vp * vp) * scale)


def dft_skycomponent_local(
    vis: xr.Dataset,
    skycomponents: list[SkyComponent],
) -> xr.Dataset:
    """Quick 'n dirty numpy-based predict for local testing without sdp.func.

    :param vis: Visibility dataset to be added to.
    :return: Accumulated Visibility dataset.
    """
    if isinstance(skycomponents, SkyComponent):
        skycomponents = [skycomponents]

    # Pre-multiply uvw by 1/lambda. Should be a small band and only a test fn.
    frequency = vis.frequency.data
    uvw = np.einsum(
        "tbd,f->tbfd",
        vis.uvw.data,
        frequency / const.c.value,  # pylint: disable=no-member
    )
    u = uvw[..., 0]
    v = uvw[..., 1]
    w = uvw[..., 2]

    # Get coordaintes of phase centre
    ra0 = vis.phasecentre.ra.radian
    cdec0 = np.cos(vis.phasecentre.dec.radian)
    sdec0 = np.sin(vis.phasecentre.dec.radian)

    # Loop over LSM components and accumulate model visibilities
    for comp in skycomponents:
        if not np.all(comp.frequency == frequency):
            raise ValueError("Only supporting equal frequency axes")
        cdec = np.cos(comp.direction.dec.radian)
        sdec = np.sin(comp.direction.dec.radian)
        cdra = np.cos(comp.direction.ra.radian - ra0)
        l_comp = cdec * np.sin(comp.direction.ra.radian - ra0)
        m_comp = sdec * cdec0 - cdec * sdec0 * cdra
        n_comp = sdec * sdec0 + cdec * cdec0 * cdra
        # Should this exponent be positive?
        #  - Compare with dft_skycomponent_visibility...
        comp_data = np.einsum(
            "tbf,fp->tbfp",
            np.exp(-2j * np.pi * (u * l_comp + v * m_comp + w * (n_comp - 1))),
            comp.flux,
        )
        if comp.shape == "GAUSSIAN":
            comp_data *= gaussian_tapers(uvw, comp.params)[..., np.newaxis]
        vis.vis.data += comp_data

    return vis


def predict_from_components(
    vis: xr.Dataset,
    skycomponents: list[SkyComponent],
    reset_vis: bool = False,
    beam_type: str = "everybeam",
    eb_coeffs: str = None,
    eb_ms: str = None,
) -> xr.Dataset:
    """Predict model visibilities from a SkyComponent List.

    :param vis: Visibility dataset to be added to.
    :param skycomponents: SkyComponent List containing the local sky model
    :param reset_vis: Whether or not to set visibilities to zero before
        accumulating components. Default is False.
    :param beam_type: Type of beam model to use. Default is "everybeam". If set
        to None, no beam will be applied.
    :param eb_coeffs: Everybeam coeffs datadir containing beam coefficients.
        Required if beam_type is "everybeam".
    :param eb_ms: Measurement set need to initialise the everybeam telescope.
        Required if bbeam_type is "everybeam".
    """
    if not isinstance(vis, xr.Dataset):
        raise ValueError(f"vis is not of type xr.Dataset: {type(vis)}")

    if len(skycomponents) == 0:
        logger.warning("No sky model components to predict")
        return

    if reset_vis:
        vis.vis.data = np.zeros(vis.vis.shape, "complex")

    # Just use the beam near the middle of the scan?
    time = np.mean(Time(vis.datetime.data))

    # Set up the beam model
    if beam_type == "everybeam":
        # Could do this once externally, but don't want to pass around
        # exotic data types.
        os.environ["EVERYBEAM_DATADIR"] = eb_coeffs
        beams = GenericBeams(vis=vis, array="Low", ms_path=eb_ms)

        # Update ITRF coordinates of the beam and normalisation factors
        beams.update_beam(frequency=vis.frequency.data, time=time)

        # Check beam pointing direction
        altaz = beams.beam_direction.transform_to(
            AltAz(obstime=time, location=beams.array_location)
        )
        if altaz.alt.degree < 0:
            raise ValueError(f"Pointing below horizon el={altaz.alt.degree}")

    else:
        response = np.zeros(
            (len(vis.configuration.id), len(vis.frequency), 2, 2),
            dtype=vis.vis.dtype,
        )
        response[..., :, :] = np.eye(2)

    # Use a less-efficient DFT if ska-sdp-func is not available
    have_sdp_func = importlib.util.find_spec("ska_sdp_func") is not None

    for comp in skycomponents:

        # Predict model visibilities for component
        compvis = vis.assign({"vis": xr.zeros_like(vis.vis)})
        if have_sdp_func:
            dft_skycomponent_visibility(compvis, comp)
        else:
            dft_skycomponent_local(compvis, comp)

        # Apply beam distortions and add to combined model visibilities
        if beam_type == "everybeam":
            # Check component direction
            altaz = comp.direction.transform_to(
                AltAz(obstime=time, location=beams.array_location)
            )
            if altaz.alt.degree < 0:
                logger.warning("LSM component [%s] below horizon", comp.name)
                continue

            response = beams.array_response(
                direction=comp.direction,
                frequency=vis.frequency.data,
                time=time,
            )

        # For a direct call, this should be an in-place update.
        # When called via xr.map_blocks, it is read-only and the
        # in-place update is handled elsewhere. There will be a cleaner
        # way to do this, but for now just get it working.
        if isinstance(vis, Visibility):
            # Assumed to be direct call
            vis.vis.data += (
                np.einsum(  # pylint: disable=too-many-function-args
                    "bfpx,tbfxy,bfqy->tbfpq",
                    response[compvis.antenna1.data, :, :, :],
                    compvis.vis.data.reshape(vis.vis.shape[:3] + (2, 2)),
                    response[compvis.antenna2.data, :, :, :].conj(),
                ).reshape(vis.vis.shape)
            )
        else:
            # Assumed to be via xr.map_blocks
            vis.vis.data = vis.vis.data + (
                np.einsum(  # pylint: disable=too-many-function-args
                    "bfpx,tbfxy,bfqy->tbfpq",
                    response[compvis.antenna1.data, :, :, :],
                    compvis.vis.data.reshape(vis.vis.shape[:3] + (2, 2)),
                    response[compvis.antenna2.data, :, :, :].conj(),
                ).reshape(vis.vis.shape)
            )

    return vis
