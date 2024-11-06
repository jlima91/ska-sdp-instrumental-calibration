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
    """Class for LSM components."""

    name: str
    RAdeg: float
    DEdeg: float
    awide: float
    bwide: float
    pawide: float
    Fint200: float
    alpha: float


def deconvolve_gaussian(a0, b0, pa0_deg):
    """Deconvolve MWA synthesised beam from Gaussian shape parameters.

    Todo. Will follow the approach from askap-analysis class MathsUtils.

    :param a0: major axis FWHM (arcseconds)
    :param b0: minor axis FWHM (arcseconds)
    :param pa0_deg: Gaussian position angle (degrees)
    :return: tuple of deconvolved parameters (same units as input)
    """
    return a0, b0, pa0_deg


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

        # Deconvolve MWA synthesised beam from Gaussian shape parameters.
        bmaj, bmin, bpa = deconvolve_gaussian(
            comp.awide, comp.bwide, comp.pawide
        )

        # pylint: disable=no-member, duplicate-code
        skycomponents.append(
            SkyComponent(
                direction=SkyCoord(
                    ra=comp.RAdeg,
                    dec=comp.DEdeg,
                    unit=(units.deg, units.deg),
                ),
                frequency=freq,
                name=comp.name,
                flux=flux,
                polarisation_frame=PolarisationFrame("linear"),
                # shape="Point",
                shape="Gaussian",
                # From what I can tell, all params units are degrees
                params={
                    "bmaj": bmaj / 3600.0,
                    "bmin": bmin / 3600.0,
                    "bpa": bpa,
                },
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
        model = Component
        model.name = "default"
        model.RAdeg = vis.phasecentre.ra.degree
        model.DEdeg = vis.phasecentre.dec.degree
        model.awide = 0.0
        model.bwide = 0.0
        model.pawide = 0.0
        model.Fint200 = 1.0
        model.alpha = 0.0
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
