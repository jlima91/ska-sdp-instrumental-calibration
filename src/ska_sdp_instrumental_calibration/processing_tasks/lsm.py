"""Module for generating the local sky model.

Note that these are temporary functions that will be replaced by functions that
connect to ska-sdp-global-sky-model functions.
"""

__all__ = [
    "convert_model_to_skycomponents",
    "deconvolve_gaussian",
    "generate_lsm_from_gleamegc",
]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from numpy import typing
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger("processing_tasks.lsm")


@dataclass
class Component:
    """Class for LSM components.

    Class to hold catalogue data for a component of the local sky model. For
    components with elliptical Gaussian parameters, if beam parameters are also
    supplied, the beam will be deconvolved from the component parameters. If
    the component parameters have already had the beam deconvolved, the beam
    parameters should be left at zero.

    Args:
        name (str): Name
        RAdeg (float): Right Ascension J2000 (degrees)
        DEdeg (float): Declination J2000 (degrees)
        flux (float): Flux density (Jy) at the reference frequency, ref_freq.
        ref_freq (float): Reference frequency (Hz). Default=200e6.
        alpha (float): Spectral index. Default=0
        major (float): Fitted semi-major axis (arcsec). Default=0
        minor (float): Fitted semi-minor axis (arcsec). Default=0
        pa (float): Fitted position angle (degrees). Default=0
        beam_major (float): Semi-major axis of a beam that is still convolved
            into the main component shape (arcsec). Default=0 (no beam present)
        beam_minor (float): Semi-minor axis of a beam that is still convolved
            into the main component shape (arcsec). Default=0 (no beam present)
        beam_pa (float): Position angle of a beam that is still convolved
            into the main component shape (degrees). Default=0
    """

    name: str
    RAdeg: float
    DEdeg: float
    flux: float
    ref_freq: float = 200e6
    alpha: float = 0.0
    major: float = 0.0
    minor: float = 0.0
    pa: float = 0.0
    beam_major: float = 0.0
    beam_minor: float = 0.0
    beam_pa: float = 0.0


def generate_lsm_from_gleamegc(
    gleamfile: str,
    phasecentre: SkyCoord,
    fov: float = 5.0,
    flux_limit: float = 0.0,
    alpha0: float = -0.78,
) -> list[Component]:
    """
    Generate a local sky model using gleamegc.

    Form a list of Component objects. If the catalogue file cannot be found, a
    single point source will be generated at phasecentre.

    All components are Gaussians with data from the following GLEAM columns.
    Where available, Fintfit200 and alpha are used to set the spectral
    parameters flux and alpha. Otherwise, Fintwide and alpha0 are used.

    See data class :class:`~Component` and the gleamegc ReadMe file for more
    information on columns extracted from the catalogue.

    :param cat: Catalogue filename. Must follow the gleamegc.dat format.
    :param phasecentre: astropy SkyCoord for the centre of the sky model.
    :param fov: Field of view in degrees. Default is 5.
    :param flux_limit: minimum flux density in Jy. Default is 0.
    :param alpha0: Nominal alpha value to use when fitted data are unspecified.
        Default is -0.78.
    :return: Component list
    """
    if not Path(gleamfile).is_file():
        logger.warn(
            f"Cannot open gleam catalogue file {gleamfile}. "
            "Returning point source with unit flux at phase centre."
        )
        return [
            Component(
                name="default",
                RAdeg=phasecentre.ra.degree,
                DEdeg=phasecentre.dec.degree,
                flux=1.0,
            )
        ]

    deg2rad = np.pi / 180.0
    max_separation = fov / 2 * deg2rad

    ra0 = phasecentre.ra.radian
    dec0 = phasecentre.dec.radian
    cosdec0 = np.cos(dec0)
    sindec0 = np.sin(dec0)

    # Model is a list of components
    model = []

    with open(gleamfile, "r") as f:

        alpha_cat = []

        for line in f:

            name = line[6:20]
            Fintwide = float(line[129:139])

            if Fintwide > flux_limit:

                ra = float(line[65:75]) * deg2rad
                dec = float(line[87:97]) * deg2rad
                separation = np.arccos(
                    np.sin(dec) * sindec0
                    + np.cos(dec) * cosdec0 * np.cos(ra - ra0)
                )
                if separation > max_separation:
                    continue

                Fintfit200 = line[3135:3145]
                if Fintfit200.strip() == "---":
                    flux = Fintwide
                    alpha = alpha0
                else:
                    flux = float(Fintfit200)
                    alpha = float(line[3104:3113])
                    alpha_cat.append(alpha)

                model.append(
                    Component(
                        name=name,
                        flux=flux,
                        ref_freq=200e6,
                        alpha=alpha,
                        RAdeg=float(line[65:75]),
                        DEdeg=float(line[87:97]),
                        major=float(line[153:165]),
                        minor=float(line[179:187]),
                        pa=float(line[200:210]),
                        beam_major=float(line[247:254]),
                        beam_minor=float(line[255:262]),
                        beam_pa=float(line[263:273]),
                    )
                )

        f.close()

        logger.info(f"extracted {len(model)} GLEAM components")
        logger.debug(
            f"alpha: mean = {np.mean(alpha_cat):.2f}, "
            + f"median = {np.median(alpha_cat):.2f}, "
            + f"used = {alpha0:.2f}"
        )

    return model


def convert_model_to_skycomponents(
    model: list[Component],
    freq: typing.NDArray[float],
) -> list[SkyComponent]:
    """Convert the LocalSkyModel to a list of SkyComponents.

    All sources are unpolarised and specified in the linear polarisation frame
    using XX = YY = Stokes I/2.

    Function :func:`~deconvolve_gaussian` is used to deconvolve the MWA
    synthesised beam from catalogue shape parameters of each component.
    Components with non-zero widths after this process are stored with
    shape = "GAUSSIAN". Otherwise shape = "POINT".

    :param model: Component list
    :param freq: Frequency list in Hz
    :param freq0: Reference Frequency for flux scaling in Hz. Default is 200e6.
        Note: freq0 should really be part of the sky model
    :return: SkyComponent list
    """
    skycomponents = []
    freq = np.array(freq)

    for comp in model:
        flux0 = comp.flux
        freq0 = comp.ref_freq
        alpha = comp.alpha

        # assume 4 pols
        flux = np.zeros((len(freq), 4))
        flux[:, 0] = flux[:, 3] = flux0 / 2 * (freq / freq0) ** alpha

        # Deconvolve synthesised beam from fitted shape parameters.
        smaj, smin, spa = deconvolve_gaussian(comp)
        if smaj == 0 and smin == 0:
            shape = "POINT"
            params = {}
        else:
            shape = "GAUSSIAN"
            # From what I can tell, all params units are degrees
            params = {
                "bmaj": smaj / 3600.0,
                "bmin": smin / 3600.0,
                "bpa": spa,
            }

        skycomponents.append(
            SkyComponent(
                direction=SkyCoord(
                    ra=comp.RAdeg,
                    dec=comp.DEdeg,
                    unit="deg",
                ),
                frequency=freq,
                name=comp.name,
                flux=flux,
                polarisation_frame=PolarisationFrame("linear"),
                shape=shape,
                params=params,
            )
        )

    return skycomponents


def deconvolve_gaussian(comp: Component) -> tuple[float]:
    """Deconvolve MWA synthesised beam from Gaussian shape parameters.

    This follows the approach of the analysisutilities function
    deconvolveGaussian in the askap-analysis repository, written by Matthew
    Whiting. This is based on the approach described in Wild (1970), AuJPh 23,
    113.

    :param comp: :class:`~Component` data for a source
    :return: Tuple of deconvolved parameters (same units as data in comp)
    """

    # fitted data on source
    fmajsq = comp.major * comp.major
    fminsq = comp.minor * comp.minor
    fdiff = fmajsq - fminsq
    fphi = 2.0 * comp.pa * np.pi / 180.0

    # beam data at source location
    bmajsq = comp.beam_major * comp.beam_major
    bminsq = comp.beam_minor * comp.beam_minor
    bdiff = bmajsq - bminsq
    bphi = 2.0 * comp.beam_pa * np.pi / 180.0

    # source data after deconvolution
    if fdiff < 1e-6:
        # Circular Gaussian case
        smaj = np.sqrt(fmajsq - bminsq)
        smin = np.sqrt(fmajsq - bmajsq)
        psmaj = np.pi / 2.0 + comp.beam_pa

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
        smaj = 0 if smajsq <= 0 else np.sqrt(smajsq)
        smin = 0 if sminsq <= 0 else np.sqrt(sminsq)
        psmaj = 0 if cossphi == 0 else np.arctan2(sinsphi, cossphi) / 2.0

    return max(smaj, smin, 0), max(min(smaj, smin), 0), psmaj * 180 / np.pi
