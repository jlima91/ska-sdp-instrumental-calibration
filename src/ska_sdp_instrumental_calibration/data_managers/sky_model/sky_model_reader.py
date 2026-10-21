import logging
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.sky_model import (
    LocalSkyModel,
    SkyComponent,
    SkyModel,
    SkyModelMetadata,
)

# NOTE: _make_fov_and_flux_filter is private to ska_sdp_datamodels, so it may
# change or disappear in a future release without notice. It is reused here so
# that GLEAM and LSM CSV reads apply the same FoV filter.
from ska_sdp_datamodels.sky_model.sky_model_io import _make_fov_and_flux_filter

from .sky_model_operations import deconvolve_gaussian

logger = logging.getLogger(__name__)

# Hurley-Walker et al. (2017), MNRAS 464, 1146
GLEAMEGC_REFERENCE = "https://doi.org/10.1093/mnras/stw2337"


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

    @staticmethod
    def df_to_components(df: pd.DataFrame) -> list[SkyComponent]:
        """
        Convert a pandas DataFrame into a list of SkyComponent objects.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe where each row represents a component. The
            column names must match the ``SkyComponent`` fields.

        Returns
        -------
        list of SkyComponent
            A list containing the converted SkyComponent instances.
        """
        return [SkyComponent(**row) for row in df.to_dict("records")]


def export_lsm_to_csv(components: list[SkyComponent], csv_path: str) -> None:
    """
    Export a list of SkyComponent instances to a CSV file.

    Parameters
    ----------
    components : list[SkyComponent]
        A list of sky components.
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
    for row, sky_comp in enumerate(components):
        local_sky_model.set_row(row, asdict(sky_comp))

    # .save method will fail with provide just file name like "test.csv"
    # it should be "./test.csv"
    # To solve this, we can use os.path.abspath(csv_path)
    local_sky_model.save(csv_path)


def generate_lsm_from_csv(
    csvfile: str,
    phasecentre: SkyCoord,
    fov: float = 5.0,
    flux_limit: float = 0.0,
) -> list[SkyComponent]:
    """
    Generate a local sky model using a CSV file.

    This function reads a CSV file and converts it into a list of
    :class:`~SkyComponent` objects. If the CSV file cannot be found, the
    function raises a ValueError.

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
    list[SkyComponent]
        A list of sky components extracted from the CSV file.

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
) -> list[SkyComponent]:
    """
    Generate a local sky model using the GLEAM-EGC catalogue.

    This is a wrapper around :func:`read_gleamegc` that returns a list of
    :class:`~SkyComponent` objects. If the file cannot be found, a warning is
    logged and a single point source at the phase centre is returned.

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
    list[SkyComponent]
        A list of sky components extracted from the catalogue.
    """
    if not Path(gleamfile).is_file():
        logger.warning(
            f"Cannot open gleam catalogue file {gleamfile}. "
            "Returning point source with unit flux at phase centre."
        )
        return [
            SkyComponent(
                component_id="default",
                source_id="",
                ra_deg=phasecentre.ra.degree,
                dec_deg=phasecentre.dec.degree,
                i_pol_jy=1.0,
                ref_freq_hz=200e6,
                epoch=0.0,
                a_arcsec=0.0,
                b_arcsec=0.0,
                pa_deg=0.0,
                spec_idx=[0.0],
                log_spec_idx=True,
            )
        ]

    sky_model = read_gleamegc(
        gleamfile,
        phase_centre=phasecentre,
        fov_deg=fov,
        flux_limit=flux_limit,
        alpha0=alpha0,
    )

    return list(sky_model.components.values())


def read_gleamegc(
    gleamfile: str,
    phase_centre: Optional[SkyCoord] = None,
    fov_deg: Optional[float] = None,
    flux_limit: float = 0.0,
    alpha0: float = -0.78,
) -> SkyModel:
    """
    Read the GLEAM-EGC catalogue into a :class:`SkyModel`.

    All components are treated as Gaussians. The function attempts to use
    'Fintfit200' and 'alpha' columns for flux and spectral index. If these are
    unavailable (marked as '---'), it falls back to 'Fintwide' and ``alpha0``.

    The GLEAM-EGC shape columns (``a_wide``, ``b_wide``, ``psf_a_wide`` and
    ``psf_b_wide``) are described as "semi-major" / "semi-minor" axes, but
    actually hold the full width at half maximum (FWHM), as produced by the
    Aegean source finder. For example, ``psf_a_wide`` is ~130 arcsec near
    zenith, matching the ~2 arcmin resolution of GLEAM at 200 MHz. They are
    halved here so that the returned components hold literal semi-axes, as
    in the GSM schema. Position angles are measured from North through East
    in both GLEAM and the GSM, and are used unchanged.

    Parameters
    ----------
    gleamfile : str
        The path to the catalogue file. Must follow the GLEAM-EGC data format.
    phase_centre : SkyCoord, optional
        Centre of the field of view. Required if ``fov_deg`` is set.
    fov_deg : float, optional
        The field of view diameter in degrees. Components further than
        ``fov_deg / 2`` from ``phase_centre`` are excluded. If None, no
        positional filtering is applied.
    flux_limit : float, optional
        The minimum wide-band flux density (``Fintwide``) in Jy. Components
        at or below this limit are excluded. Default is 0.0.
    alpha0 : float, optional
        The nominal spectral index to use when fitted data are unspecified in
        the catalogue. Default is -0.78.

    Returns
    -------
    SkyModel
        Sky model with the GLEAM-EGC components and catalogue metadata.

    Raises
    ------
    ValueError
        If ``fov_deg`` is set without ``phase_centre``.
    FileNotFoundError
        If ``gleamfile`` does not exist.
    """
    # The flux cut stays on Fintwide below, so only the FoV filter is used.
    # Raises ValueError if fov_deg is set without phase_centre.
    accept_position = _make_fov_and_flux_filter(
        ["ra_deg", "dec_deg"], phase_centre, fov_deg, min_flux_jy=None
    )

    components = {}

    with open(gleamfile, "r") as f:

        alpha_cat = []

        for line in f:

            name = line[6:20]
            Fintwide = float(line[129:139])

            if Fintwide > flux_limit:

                ra_token = line[65:75]
                dec_token = line[87:97]

                if accept_position is not None and not accept_position(
                    [ra_token, dec_token]
                ):
                    continue

                Fintfit200 = line[3135:3145]
                if Fintfit200.strip() == "---":
                    flux = Fintwide
                    alpha = alpha0
                else:
                    flux = float(Fintfit200)
                    alpha = float(line[3104:3113])
                    alpha_cat.append(alpha)

                # Deconvolve synthesised beam from fitted shape parameters.
                # GLEAM axes are FWHM, so halve them to get semi-axes.
                major_ax, minor_ax, pos_ang = deconvolve_gaussian(
                    float(line[153:165]) / 2,
                    float(line[179:187]) / 2,
                    float(line[200:210]),
                    float(line[247:254]) / 2,
                    float(line[255:262]) / 2,
                    float(line[263:273]),
                )

                components[name] = SkyComponent(
                    component_id=name,
                    source_id="",
                    ra_deg=float(ra_token),
                    dec_deg=float(dec_token),
                    i_pol_jy=flux,
                    ref_freq_hz=200e6,
                    epoch=0.0,
                    a_arcsec=major_ax,
                    b_arcsec=minor_ax,
                    pa_deg=pos_ang,
                    spec_idx=[alpha],
                    log_spec_idx=True,
                )

        f.close()

        logger.info(f"extracted {len(components)} GLEAM components")
        logger.debug(
            f"alpha: mean = {np.mean(alpha_cat):.2f}, "
            + f"median = {np.median(alpha_cat):.2f}, "
            + f"used = {alpha0:.2f}"
        )

    metadata = SkyModelMetadata(
        version="1.0",
        catalogue_name="GLEAM-EGC",
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        reference=GLEAMEGC_REFERENCE,
        freq_min_hz=72e6,
        freq_max_hz=231e6,
    )

    return SkyModel(metadata=metadata, components=components)


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
