import functools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .component import Component

logger = logging.getLogger(__name__)


SKY_MODEL_CSV_HEADER = [
    "component_id",
    "ra",
    "dec",
    "i_pol",
    "major_ax",
    "minor_ax",
    "pos_ang",
    "ref_freq",
    "spec_idx",
    "log_spec_idx",
]


class ComponentConverters:
    __headers_to_fields_pairs = [
        ("name", "component_id"),
        ("RAdeg", "ra"),
        ("DEdeg", "dec"),
        ("flux", "i_pol"),
        ("major", "major_ax"),
        ("minor", "minor_ax"),
        ("pa", "pos_ang"),
        ("ref_freq", "ref_freq"),
        ("alpha", "spec_idx"),
        ("log_spec_idx", "log_spec_idx"),
    ]

    __exponent_str = functools.partial(lambda value: format(value, "e"))
    __six_decimal_str = functools.partial(lambda value: format(value, ".6f"))
    _list_str = functools.partial(
        lambda value: json.dumps(value) if value is not None else '"[]"'
    )

    __headers_formatter = {
        "component_id": str,
        "ra": str,
        "dec": str,
        "i_pol": __exponent_str,
        "major_ax": __exponent_str,
        "minor_ax": __exponent_str,
        "pos_ang": __six_decimal_str,
        "ref_freq": __exponent_str,
        "spec_idx": _list_str,
        "log_spec_idx": str,
    }

    __non_existing_field_default = 0.0

    @classmethod
    def to_csv_row(cls, component: Component) -> list[str]:
        return [
            cls.__headers_formatter[header](
                getattr(
                    component,
                    obj_prop,
                    cls.__non_existing_field_default,
                )
                if obj_prop
                else cls.__non_existing_field_default
            )
            for obj_prop, header in cls.__headers_to_fields_pairs
            if header in SKY_MODEL_CSV_HEADER
        ]

    @classmethod
    def row_to_component(cls, row: pd.Series) -> Component:
        """
        Convert a pandas Series representing a row into a Component object.

        Parameters
        ----------
        row : pd.Series
            A series containing keys that match the CSV header names defined
            in PROPERTY_PAIRS.

        Returns
        -------
        Component
            The instantiated component object.
        """
        kwargs = {
            object_prop: row[csv_header]
            for object_prop, csv_header in cls.__headers_to_fields_pairs
            if csv_header in row and object_prop is not None
        }
        return Component(**kwargs)

    @classmethod
    def df_to_components(cls, df: pd.DataFrame) -> list[Component]:
        """
        Convert a pandas DataFrame into a list of Component objects.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe where each row represents a component.

        Returns
        -------
        list of Component
            A list containing the converted Component instances.
        """
        return [cls.row_to_component(row) for _, row in df.iterrows()]


def generate_lsm_from_csv(
    csvfile: str,
    phasecentre: SkyCoord,
    fov: float = 5.0,
    flux_limit: float = 0.0,
) -> list[Component]:
    """
    Generate a local sky model using a CSV file.

    This function reads a CSV file and converts it into a list of
    :class:`~Component` objects. If the CSV file cannot be found, the function
    raises a ValueError.

    The CSV file is expected to have a commented line header and the following
    10 columns in order:
    component_id, ra, dec, i_pol, major_ax, minor_ax, pos_ang, ref_freq,
    spec_idx, log_spec_idx.

    The spec_idx column should contain a JSON array of spectral indices
    (e.g., "[-0.7,0.01,0.123]"). The log_spec_idx column is a boolean
    indicating whether the spectral index was calculated using a logarithmic
    (true) or linear (false) model.

    All components are treated as Gaussians, though many may default to point
    sources.

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
        csvfile,
        sep=",",
        comment="#",
        names=headers,
        skipinitialspace=True,
        converters={
            "spec_idx": lambda x: json.loads(x) if x else None,
            "log_spec_idx": lambda x: x.lower() == "true" if x else True,
        },
    )

    lsm_df = lsm_df[lsm_df["i_pol"] >= flux_limit]

    lsm_df["ra_radian"] = deg2rad * lsm_df["ra"]
    lsm_df["dec_radian"] = deg2rad * lsm_df["dec"]
    lsm_df["separation"] = np.arccos(
        np.sin(lsm_df["dec_radian"]) * sindec0
        + np.cos(lsm_df["dec_radian"])
        * cosdec0
        * np.cos(lsm_df["ra_radian"] - ra0)
    )

    lsm_df = lsm_df[lsm_df["separation"] <= max_separation]

    model = ComponentConverters.df_to_components(lsm_df)

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
                        alpha=[alpha],
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
