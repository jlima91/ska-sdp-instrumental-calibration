import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .component import Component

logger = logging.getLogger(__name__)


SKY_MODEL_CSV_HEADER = [
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


class ComponentConverters:
    __headers_to_fields_map = {
        "RA (deg)": "RAdeg",
        "Dec (deg)": "DEdeg",
        "I (Jy)": "flux",
        "Q (Jy)": "doesnotexist",
        "U (Jy)": "doesnotexist",
        "V (Jy)": "doesnotexist",
        "Ref. freq. (Hz)": "ref_freq",
        "Spectral index": "alpha",
        "Rotation measure (rad/m^2)": "doesnotexist",
        "FWHM major (arcsec)": "major",
        "FWHM minor (arcsec)": "minor",
        "Position angle (deg)": "pa",
    }

    __exponent_str = functools.partial(lambda value: format(value, "e"))
    __six_decimal_str = functools.partial(lambda value: format(value, ".6f"))

    __headers_formatter = {
        "RA (deg)": str,
        "Dec (deg)": str,
        "I (Jy)": __exponent_str,
        "Q (Jy)": __exponent_str,
        "U (Jy)": __exponent_str,
        "V (Jy)": __exponent_str,
        "Ref. freq. (Hz)": __exponent_str,
        "Spectral index": __exponent_str,
        "Rotation measure (rad/m^2)": __exponent_str,
        "FWHM major (arcsec)": __exponent_str,
        "FWHM minor (arcsec)": __exponent_str,
        "Position angle (deg)": __six_decimal_str,
    }

    __non_existing_field_default = 0.0

    @classmethod
    def to_csv_row(cls, component: Component) -> list[str]:
        row = []
        for header in SKY_MODEL_CSV_HEADER:
            formatter = cls.__headers_formatter[header]
            row.append(
                formatter(
                    getattr(
                        component,
                        cls.__headers_to_fields_map[header],
                        cls.__non_existing_field_default,
                    )
                )
            )
        return row


def generate_lsm_from_csv(
    csvfile: str,
    phasecentre: SkyCoord,
    fov: float = 5.0,
    flux_limit: float = 0.0,
) -> list[Component]:
    """
    Generate a local sky model using a CSV file.

    This function reads a CSV file formatted for OSKAR simulations and converts
    it into a list of :class:`~Component` objects. If the CSV file cannot be
    found, the function raises a ValueError.

    The CSV file is expected to have a mandatory header row and the following
    12 columns in order:
    RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz),
    Spectral index, Rotation measure (rad/m^2), FWHM major (arcsec),
    FWHM minor (arcsec), Position angle (deg).

    Frequency parameters (Q, U, V, and RM) are ignored. All components are
    treated as Gaussians, though many may default to point sources.

    Parameters
    ----------
    csvfile : str
        The path to the CSV file. Must follow the format described above.
    phasecentre : SkyCoord
        The phase centre of the observation, serving as the reference point for
        the sky model.
    fov : float, optional
        The field of view diameter in degrees. Components outside this radius
        (centered on `phasecentre`) are excluded. Default is 5.0.
    flux_limit : float, optional
        The minimum flux density in Jy. Components below this limit are
        excluded. Default is 0.0.

    Returns
    -------
    list[Component]
        A list of component objects extracted from the CSV file.

    Raises
    ------
    ValueError
        If the `csvfile` does not exist or cannot be opened.
    """
    if not Path(csvfile).is_file():
        raise ValueError(f"Cannot open csv file {csvfile}")

    deg2rad = np.pi / 180.0
    max_separation = fov / 2 * deg2rad
    ra0 = phasecentre.ra.radian
    dec0 = phasecentre.dec.radian
    cosdec0 = np.cos(dec0)
    sindec0 = np.sin(dec0)

    headers = SKY_MODEL_CSV_HEADER

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
    Generate a local sky model using the GLEAM-EGC catalogue.

    This function parses a GLEAM-EGC formatted file and converts it into a list
    of :class:`~Component` objects. If the file cannot be found, a warning is
    logged and a single point source at the phase centre is returned.

    All components are treated as Gaussians. The function attempts to use
    'Fintfit200' and 'alpha' columns for flux and spectral index. If these are
    unavailable (marked as '---'), it falls back to 'Fintwide' and ``alpha0``.

    Parameters
    ----------
    gleamfile : str
        The path to the catalogue file. Must follow the GLEAM-EGC data format.
    phasecentre : SkyCoord
        The phase centre of the observation, serving as the reference point for
        the sky model.
    fov : float, optional
        The field of view diameter in degrees. Components outside this radius
        (centered on ``phasecentre``) are excluded. Default is 5.0.
    flux_limit : float, optional
        The minimum flux density in Jy. Components below this limit are
        excluded. Default is 0.0.
    alpha0 : float, optional
        The nominal spectral index to use when fitted data are unspecified in
        the catalogue. Default is -0.78.

    Returns
    -------
    list[Component]
        A list of component objects extracted from the catalogue.
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
