import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.global_sky_model import LocalSkyModel, SkyComponent

from .component import Component

logger = logging.getLogger(__name__)


class ComponentConverters:
    @staticmethod
    def create_lsm_df(local_sky_model: LocalSkyModel) -> pd.DataFrame:
        """
        Create a pandas DataFrame from a LocalSkyModel instance.

        Parameters
        ----------
        local_sky_model : LocalSkyModel
            The local sky model object containing astronomical component
            data.

        Returns
        -------
        :class:`pandas.DataFrame`
            A DataFrame where columns correspond to the sky model's
            defined fields and rows represent individual components.
        """
        cols = local_sky_model.column_names

        components_table = zip(*[local_sky_model[col] for col in cols])

        return pd.DataFrame(components_table, columns=cols)

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

        # Creation of components from sky_components (datamodel) can be removed
        # when validation part goes to LocalSkyModel (datamodel)
        # and maybe we will entirly use Component from datamodels.

        sky_components = (SkyComponent(**row) for row in df.to_dict("records"))

        return list(
            ComponentConverters.sky_components_to_components(sky_components)
        )

    @staticmethod
    def sky_components_to_components(
        sky_components: Iterable[SkyComponent],
    ) -> Iterable[Component]:
        """
        Convert SkyComponent instances to external Component instances.

        Parameters
        ----------
        sky_components : Iterable[SkyComponent]
            An iterable of internal sky component dataclass instances.

        Returns
        -------
        Iterable[Component]
            An iterable of component objects compatible with the
            external local sky model library.
        """
        return (
            Component(
                component_id=sky_comp.component_id,
                source_id=sky_comp.source_id,
                epoch=sky_comp.epoch,
                ra=sky_comp.ra_deg,
                dec=sky_comp.dec_deg,
                i_pol=sky_comp.i_pol_jy,
                ref_freq=sky_comp.ref_freq_hz,
                spec_idx=list(sky_comp.spec_idx),
                major_ax=sky_comp.a_arcsec,
                minor_ax=sky_comp.b_arcsec,
                pos_ang=sky_comp.pa_deg,
                log_spec_idx=sky_comp.log_spec_idx,
            )
            for sky_comp in sky_components
        )

    @staticmethod
    def components_to_sky_components(
        components: Iterable[Component],
    ) -> Iterable[SkyComponent]:
        """
        Convert external Component instances to SkyComponent instances.

        Parameters
        ----------
        components : Iterable[Component]
            An iterable of component objects from the external local sky
            model library.

        Returns
        -------
        Iterable[SkyComponent]
            An iterable of internal sky component dataclass instances with
            explicit unit-suffixed fields.
        """
        return (
            SkyComponent(
                component_id=comp.component_id,
                source_id=comp.source_id,
                epoch=comp.epoch,
                ra_deg=comp.ra,
                dec_deg=comp.dec,
                i_pol_jy=comp.i_pol,
                ref_freq_hz=comp.ref_freq,
                a_arcsec=comp.major_ax,
                b_arcsec=comp.minor_ax,
                pa_deg=comp.pos_ang,
                spec_idx=comp.spec_idx,
                log_spec_idx=comp.log_spec_idx,
            )
            for comp in components
        )


def export_lsm_to_csv(components: list[Component], csv_path: str) -> None:
    """
    Export a list of Component instances to a CSV file.

    Parameters
    ----------
    components : list[Component]
        A list of component objects from the external library.
    csv_path : str
        The destination path for the CSV file.

    Returns
    -------
    None
    """
    column_names = [field.name for field in fields(SkyComponent)]
    vector_columns = ["spec_idx"]

    local_sky_model = LocalSkyModel(
        column_names=column_names,
        num_rows=len(components),
        vector_columns=vector_columns,
    )
    for row, sky_com in enumerate(
        ComponentConverters.components_to_sky_components(components)
    ):
        local_sky_model.set_row(row, asdict(sky_com))

    # .save method will fail with provide just file name like "test.csv"
    # it should be "./test.csv"
    local_sky_model.save(csv_path)


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

    local_sky_model = LocalSkyModel.load(csvfile)
    lsm_df = ComponentConverters.create_lsm_df(local_sky_model)

    lsm_df = _filter_by_fov_and_flux(lsm_df, phasecentre, fov, flux_limit)

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
                component_id="default",
                ra=phasecentre.ra.degree,
                dec=phasecentre.dec.degree,
                i_pol=1.0,
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
                        component_id=name,
                        i_pol=flux,
                        ref_freq=200e6,
                        spec_idx=[alpha],
                        ra=float(line[65:75]),
                        dec=float(line[87:97]),
                        major_ax=float(line[153:165]),
                        minor_ax=float(line[179:187]),
                        pos_ang=float(line[200:210]),
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


def _filter_by_fov_and_flux(df, phasecentre, fov, flux_limit):
    deg2rad = np.pi / 180.0
    max_separation = fov / 2 * deg2rad

    df = df[df["i_pol_jy"] >= flux_limit]

    ra_rad = deg2rad * df["ra_deg"]
    dec_rad = deg2rad * df["dec_deg"]
    separation = _calculate_separation(
        ra_rad, dec_rad, phasecentre.ra.radian, phasecentre.dec.radian
    )

    return df[separation <= max_separation]


def _calculate_separation(ra, dec, ra0, dec0):
    return np.arccos(
        np.sin(dec) * np.sin(dec0)
        + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    )


def _to_float(x):
    return float(x) if x and x.strip() else None
