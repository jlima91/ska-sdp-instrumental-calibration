"""Module for generating the local sky model.

Note that these are temporary functions that will be replaced by functions that
connect to ska-sdp-global-sky-model functions.
"""

__all__ = [
    "generate_lsm",
]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Component:
    """Class for LSM components.

    Class to hold the relevant columns of GLEAMEGC for a component of the
    local sky model.

    Args:
        name (str): GLEAM name (JHHMMSS+DDMMSS)
        RAdeg (float): Right Ascension J2000 (degrees)
        DEdeg (float): Declination J2000 (degrees)
        awide (float): Fitted semi-major axis in wide-band image (arcsec)
        bwide (float): Fitted semi-minor axis in wide-band image (arcsec)
        pawide (float): Fitted position angle in wide-band image (degrees)
        psfawide (float): Semi-major axis of the PSF (arcsec)
        psfbwide (float): Semi-minor axis of the PSF (arcsec)
        psfpawide (float): Position angle of the PSF (degrees)
        Fint200 (float): 200MHz integrated flux density
        alpha (float): Spectral index
    """

    name: str
    RAdeg: float
    DEdeg: float
    awide: float
    bwide: float
    pawide: float
    psfawide: float
    psfbwide: float
    psfpawide: float
    Fint200: float
    alpha: float


def deconvolve_gaussian(comp):
    """Deconvolve MWA synthesised beam from Gaussian shape parameters.

    This follows the approach of the analysisutilities function
    deconvolveGaussian in the askap-analysis repository, written by Matthew
    Whiting <matthew.whiting@csiro.au>. This is based on the approach described
    in Wild (1970), AuJPh 23, 113.

    :param comp: Component data for a source
    :return: Tuple of deconvolved parameters (same units as data in comp)
    """

    # fitted data on source
    fmajsq = comp.awide * comp.awide
    fminsq = comp.bwide * comp.bwide
    fdiff = fmajsq - fminsq
    fphi = 2.0 * comp.pawide * np.pi / 180.0

    # beam data at source location
    bmajsq = comp.psfawide * comp.psfawide
    bminsq = comp.psfbwide * comp.psfbwide
    bdiff = bmajsq - bminsq
    bphi = 2.0 * comp.psfpawide * np.pi / 180.0

    # source data after deconvolution
    if fdiff < 1e-6:
        # Circular Gaussian case
        smaj = np.sqrt(fmajsq - bminsq)
        smin = np.sqrt(fmajsq - bmajsq)
        psmaj = np.pi / 2.0 + comp.psfpawide

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


def convert_model_to_skycomponents(model, freq, freq0=200e6):
    """Convert the LocalSkyModel to a list of SkyComponents.

    Note: freq0 should really be part of the sky model

    All sources are unpolarised and specified in the linear polarisation frame
    using XX = YY = Stokes I / 2.

    :param : Component list
    :param : Frequency list in Hz
    :param : Reference Frequency for flux scaling is 200 MHz
    :return: SkyComponent list
    """
    skycomponents = []
    freq = np.array(freq)

    for comp in model:
        alpha = comp.alpha
        flux0 = comp.Fint200

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
                    unit=(units.deg, units.deg),  # pylint: disable=no-member
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


def generate_lsm(gleamfile, vis, fov=5.0, flux_limit=0.0, alpha0=-0.78):
    """
    Generate a local sky model using gleamegc.

    Form a list of SkyComponent objects.
    All components are Gaussians with data from the following GLEAM columns.
    Fintfit200 and alpha are used to set flux density spectra where available.
    Otherwise, Fintwide and alpha0 are used.

    From the gleamegc ReadMe file:
       7-  20 A14    ---     GLEAM      GLEAM name (JHHMMSS+DDMMSS) (Name)
      66-  75 F10.6  deg     RAdeg      Right Ascension J2000 (RAJ2000)
      88-  97 F10.6  deg     DEdeg      Declination J2000 (DEJ2000)
     130- 139 F10.6  Jy      Fintwide   Integrated flux in wide (170-231MHz)
     154- 165 E12.6  arcsec  awide      Fitted semi-major axis in wide
     180- 187 F8.4   arcsec  bwide      Fitted semi-minor axis in wide
     201- 210 F10.6  deg     pawide     Fitted position angle in wide
     248- 254  F7.3  arcsec  psfawide   Semi-major axis of the PSF in wide
     256- 262  F7.3  arcsec  psfbwide   Semi-minor axis of the PSF in wide
     264- 273  F10.6 deg     psfPAwide  Position angle of the PSF in wide
    3105-3113 F9.6   ---     alpha      ? Fitted spectral index (alpha)
    3136-3145 F10.6  Jy     Fintfit200  ? Fitted 200MHz integrated flux density

    :param cat: Catalogue filename. Must follow the gleamegc.dat format.
    :param vis: Visibility data (used to extra metadata)
    :param fov: Field of view in degrees. Default is 5.
    :param flux_limit: minimum flux density in Jy. Default is 0.
    :param alpha0: Nominal alpha value to use when fitted data are unspecified.
        Default is -0.78.
    :return: SkyComponent list
    """
    if not Path(gleamfile).is_file():
        logger.warn(
            f"Cannot open gleam catalogue file {gleamfile}. "
            "Returning point source with unit flux at phase centre."
        )

        model = Component(
            name="default",
            RAdeg=vis.phasecentre.ra.degree,
            DEdeg=vis.phasecentre.dec.degree,
            awide=0.0,
            bwide=0.0,
            pawide=0.0,
            psfawide=0.0,
            psfbwide=0.0,
            psfpawide=0.0,
            Fint200=1.0,
            alpha=0.0,
        )

        return convert_model_to_skycomponents(
            [model], vis.frequency.data, freq0=200e6
        )

    deg2rad = np.pi / 180.0
    max_separation = fov / 2 * deg2rad

    ra0 = vis.phasecentre.ra.radian
    dec0 = vis.phasecentre.dec.radian
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
                    Fint200 = Fintwide
                    alpha = alpha0
                else:
                    Fint200 = float(Fintfit200)
                    alpha = float(line[3104:3113])
                    alpha_cat.append(alpha)

                model.append(
                    Component(
                        name=name,
                        RAdeg=float(line[65:75]),
                        DEdeg=float(line[87:97]),
                        awide=float(line[153:165]),
                        bwide=float(line[179:187]),
                        pawide=float(line[200:210]),
                        psfawide=float(line[247:254]),
                        psfbwide=float(line[255:262]),
                        psfpawide=float(line[263:273]),
                        Fint200=Fint200,
                        alpha=alpha,
                    )
                )

        f.close()

        logger.info(f"extracted {len(model)} GLEAM components")
        logger.debug(
            f"alpha: mean = {np.mean(alpha_cat):.2f}, "
            + f"median = {np.median(alpha_cat):.2f}, "
            + f"used = {alpha0:.2f}"
        )

    return convert_model_to_skycomponents(
        model, vis.frequency.data, freq0=200e6
    )
