"""Module for predicting model visibilities.
"""

__all__ = [
    "predict_from_components",
    "generate_central_beams",
    "generate_rotation_matrices",
]

import gc
import importlib
import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray as xr
from astropy import constants as const
from astropy.coordinates import AltAz
from astropy.time import Time
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.beams import (
    GenericBeams,
)

logger = setup_logger("processing_tasks.predict")


def gaussian_tapers(
    uvw: npt.NDArray[float],
    params: dict[float],
) -> npt.NDArray[float]:
    """Calculated visibility amplitude tapers for Gaussian components.

    Note: this needs to be tested. Generate, image and fit a model component?

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
        comp_data = np.einsum(
            "tbf,fp->tbfp",
            np.exp(-2j * np.pi * (u * l_comp + v * m_comp + w * (n_comp - 1))),
            comp.flux,
        )
        if comp.shape == "GAUSSIAN":
            comp_data *= gaussian_tapers(uvw, comp.params)[..., np.newaxis]
        vis.vis.data = vis.vis.data + comp_data

    return vis


def generate_rotation_matrices(
    rm: npt.NDArray[float],
    frequency: npt.NDArray[float],
    dtype: float = float,
) -> npt.NDArray[float]:
    """Generate station rotation matrices from RM values.

    :param rm: 1D array of rotation measure values [nstation].
    :param frequency: 1D array of frequency values [nfrequency].
    :return: 4D array of rotation matrices: [nstation, nfrequency, 2, 2].
    """
    rot_array = np.zeros((len(rm), len(frequency), 2, 2), dtype=dtype)
    lambda_sq = (const.c.value / frequency) ** 2  # pylint: disable=no-member
    for stn, val in enumerate(rm):
        phi = val * lambda_sq
        rot_array[stn] = np.stack(
            (np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi)),
            axis=1,
            dtype=dtype,
        ).reshape(-1, 2, 2)

    return rot_array


def predict_from_components(
    vis: xr.Dataset,
    skycomponents: list[SkyComponent],
    reset_vis: bool = False,
    beam_type: str = "everybeam",
    eb_coeffs: Optional[str] = None,
    eb_ms: Optional[str] = None,
    station_rm: Optional[npt.NDArray[float]] = None,
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
    :param station_rm: Station rotation measure values. Default is None.
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
        logger.info("Using EveryBeam model in predict")
        if eb_coeffs is None or eb_ms is None:
            raise ValueError("eb_coeffs and eb_ms required for everybeam")
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
        logger.info("No beam model used in predict")

    # Set up the Faraday rotation model
    if station_rm is not None:
        if len(station_rm) != len(vis.configuration.id):
            raise ValueError("unexpected length for station_rm")
        rot_array = generate_rotation_matrices(
            station_rm, vis.frequency.data, dtype=vis.vis.dtype
        )
    else:
        rot_array = np.zeros(
            (len(vis.configuration.id), len(vis.frequency), 2, 2),
            dtype=vis.vis.dtype,
        )
        rot_array[..., :, :] = np.eye(2)

    # Use dft_skycomponent_local when the sdp-func DFT is unavailable
    use_local_dft = importlib.util.find_spec("ska_sdp_func") is None

    if not use_local_dft:
        # The ska-sdp-func version does not taper Gaussians, so do it below
        need_uvws = False
        for comp in skycomponents:
            if comp.shape == "GAUSSIAN":
                need_uvws = True
                break
        if need_uvws:
            uvw = np.einsum(
                "tbd,f->tbfd",
                vis.uvw.data,
                vis.frequency.data
                / const.c.value,  # pylint: disable=no-member
            )

    # dataset workspace for applying component-dependent effects
    compvis = vis.assign({"vis": xr.zeros_like(vis.vis)})

    for comp in skycomponents:

        # Predict model visibilities for component
        compvis.vis.data *= 0
        if use_local_dft:
            dft_skycomponent_local(compvis, comp)
        else:
            dft_skycomponent_visibility(compvis, comp)
            if comp.shape == "GAUSSIAN":
                # Apply Gaussian tapers
                compvis.vis.data *= gaussian_tapers(uvw, comp.params)[
                    ..., np.newaxis
                ]

        # Apply beam distortions and add to combined model visibilities
        if beam_type == "everybeam":
            # Check component direction
            altaz = comp.direction.transform_to(
                AltAz(obstime=time, location=beams.array_location)
            )
            if altaz.alt.degree < 0:
                logger.warning("LSM component [%s] below horizon", comp.name)
                continue
            # This ID mapping will not always work when the eb_ms file is
            # different. Should restrict the form of the eb_ms files allowed,
            # or preferably deprecate the eb_ms option.
            response = (
                beams.array_response(
                    direction=comp.direction,
                    frequency=vis.frequency.data,
                    time=time,
                )[vis.configuration.id]
                @ rot_array
            )
        else:
            response = rot_array

        # Accumulate component in the main dataset
        vis.vis.data = vis.vis.data + (
            np.einsum(  # pylint: disable=too-many-function-args
                "bfpx,tbfxy,bfqy->tbfpq",
                response[compvis.antenna1.data, :, :, :],
                compvis.vis.data.reshape(vis.vis.shape[:3] + (2, 2)),
                response[compvis.antenna2.data, :, :, :].conj(),
            ).reshape(vis.vis.shape)
        )

    # clean up component data
    del compvis
    gc.collect()

    return vis


def generate_central_beams(
    gaintable: xr.Dataset,
    vis: xr.Dataset,
    beam_type: str = "everybeam",
    eb_coeffs: Optional[str] = None,
    eb_ms: Optional[str] = None,
) -> xr.Dataset:
    """Generate beam models used in prediction at beam centre.

    :param gain_table: GainTable dataset to update.
    :param vis: Visibility dataset.
    :param beam_type: Type of beam model to use. Default is "everybeam".
    :param eb_coeffs: Everybeam coeffs datadir containing beam coefficients.
        Required if beam_type is "everybeam".
    :param eb_ms: Measurement set need to initialise the everybeam telescope.
        Required if bbeam_type is "everybeam".
    """
    if not isinstance(gaintable, xr.Dataset):
        raise ValueError("gaintable is not of type xr.Dataset")
    if not isinstance(vis, xr.Dataset):
        raise ValueError("vis is not of type xr.Dataset")
    if np.any(gaintable.frequency.data != vis.frequency.data):
        raise ValueError("Inconsistent frequencies")
    if len(gaintable.time) != 1:
        raise ValueError("Unexpected gaintable time axis")

    # Just use the beam near the middle of the scan?
    time = np.mean(Time(vis.datetime.data))

    # Set up the beam model
    # Keep this consistent with predict_from_components
    if beam_type == "everybeam":
        logger.info("Using EveryBeam model in predict")
        if eb_coeffs is None or eb_ms is None:
            raise ValueError("eb_coeffs and eb_ms required for everybeam")
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

        response = beams.array_response(
            direction=beams.beam_direction,
            frequency=vis.frequency.data,
            time=time,
        )[vis.configuration.id]

    else:
        logger.info("No beam model used in predict")
        response = np.zeros(
            (len(vis.configuration.id), len(vis.frequency.data), 2, 2),
            dtype=gaintable.gain.dtype,
        )

    assert np.all(response.shape == gaintable.gain.data[0].shape)
    gaintable.gain.data[0] = response

    return gaintable
