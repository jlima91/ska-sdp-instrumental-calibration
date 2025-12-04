import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .component import Component

logger = logging.getLogger(__name__)


def generate_lsm_from_csv(
    csvfile: str,
    phasecentre: SkyCoord,
    fov: float = 5.0,
    flux_limit: float = 0.0,
) -> list[Component]:
    """
    Generate a local sky model using a csv file of the format used in various
    OSKAR simulations.

    Form a list of Component objects. If the csv file cannot be found, a single
    point source will be generated at phasecentre.

    The csv files are expected to have the following 12 columns:
    RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy),
    Ref. freq. (Hz), Spectral index, Rotation measure (rad/m^2),
    FWHM major (arcsec), FWHM minor (arcsec), Position angle (deg)

    The header is mandatory, and is the first uncommented line of the CSV file.

    Frequency parameters (Q, U, V and RM) are ignored in this function.

    All components are Gaussians, although many will typically default to
    point sources. See data class :class:`~Component` for more information.

    :param csvfile: CSV filename. Must follow the format above.
    :param phasecentre: astropy SkyCoord for the centre of the sky model.
    :param fov: Field of view in degrees. Default is 5.
    :param flux_limit: minimum flux density in Jy. Default is 0.
    :return: Component list
    """
    if not Path(csvfile).is_file():
        raise ValueError(f"Cannot open csv file {csvfile}")

    deg2rad = np.pi / 180.0
    max_separation = fov / 2 * deg2rad
    ra0 = phasecentre.ra.radian
    dec0 = phasecentre.dec.radian
    cosdec0 = np.cos(dec0)
    sindec0 = np.sin(dec0)

    headers = [
        "RA (deg)",
        "Dec (deg)",
        "I (Jy)",
        "Q (Jy)",
        "U (Jy)",
        "V (Jy)",
        "Ref. freq. (Hz)",
        "Spectral index",
        "Rotation measure (rad/m^2)",
        "FWHM major (arcsec)",
        "FWHM minor (arcsec)",
        "Position angle (deg)",
    ]

    lsm_df = pd.read_csv(
        csvfile, sep=",", comment="#", names=headers, dtype=float
    )

    lsm_df["comp_name"] = "comp" + lsm_df.index.astype("str")

    lsm_df = lsm_df[lsm_df["I (Jy)"] >= flux_limit]
    lsm_df["ra"] = deg2rad * lsm_df["RA (deg)"]
    lsm_df["dec"] = deg2rad * lsm_df["Dec (deg)"]
    lsm_df["separation"] = np.arccos(
        np.sin(lsm_df["dec"]) * sindec0
        + np.cos(lsm_df["dec"]) * cosdec0 * np.cos(lsm_df["ra"] - ra0)
    )

    lsm_df = lsm_df[lsm_df["separation"] <= max_separation]

    model = lsm_df.apply(
        lambda row: Component(
            name=row["comp_name"],
            RAdeg=row["RA (deg)"],
            DEdeg=row["Dec (deg)"],
            flux=row["I (Jy)"],
            ref_freq=row["Ref. freq. (Hz)"],
            alpha=row["Spectral index"],
            major=row["FWHM major (arcsec)"],
            minor=row["FWHM minor (arcsec)"],
            pa=row["Position angle (deg)"],
        ),
        axis=1,
    ).tolist()

    logger.info(f"extracted {len(model)} csv components")
    return model


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

    :param gleamfile: Catalogue filename. Must follow the gleamegc.dat format.
    :param phasecentre: astropy SkyCoord for the centre of the sky model.
    :param fov: Field of view in degrees. Default is 5.
    :param flux_limit: minimum flux density in Jy. Default is 0.
    :param alpha0: Nominal alpha value to use when fitted data are unspecified.
        Default is -0.78.
    :return: Component list
    """
    if not Path(gleamfile).is_file():
        logger.warning(
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
