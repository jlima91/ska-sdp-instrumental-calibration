import logging
from dataclasses import asdict, fields
from datetime import datetime, timezone
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

        logger.info(f"extracted {len(components)} GLEAM components")
        if alpha_cat:
            logger.debug(
                f"alpha: mean = {np.mean(alpha_cat):.2f}, "
                + f"median = {np.median(alpha_cat):.2f}, "
                + f"used = {alpha0:.2f}"
            )

    metadata = SkyModelMetadata(
        version="1.0.0",
        catalogue_name="GLEAM-EGC",
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        reference=GLEAMEGC_REFERENCE,
        freq_min_hz=72e6,
        freq_max_hz=231e6,
    )

    return SkyModel(metadata=metadata, components=components)
