"""Module for generating the local sky model.

Note that these are temporary functions that will be replaced by functions that
connect to ska-sdp-global-sky-model functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_datamodels.sky_model.sky_model_operations import (
    calculate_flux_at_freq,
)

from ..beams import convert_time_to_solution_time
from .sky_model_operations import is_above_horizon

logger = logging.getLogger(__name__)


@dataclass
class Component:
    """
    Class for LSM components.

    Class to hold catalogue data for a component of the local sky model.
    Elliptical Gaussian parameters are expected to be already deconvolved
    from any synthesised beam.
    """

    component_id: str
    "Name of the component"
    ra: float
    "Right Ascension J2000 (degrees)"
    dec: float
    "Declination J2000 (degrees)"
    i_pol: float
    "I polarization - flux at reference frequency, ref_freq"
    ref_freq: float = 200e6
    "Reference frequency (Hz)"
    spec_idx: Optional[list] = None
    "Spectral index polynomial coefficients (up to 5 terms)."
    major_ax: Optional[float] = 0.0
    "Fitted semi-major axis, i.e. half of the FWHM (arcsec) at reference "
    "frequency"
    minor_ax: Optional[float] = 0.0
    "Fitted semi-minor axis, i.e. half of the FWHM (arcsec) at reference "
    "frequency"
    pos_ang: Optional[float] = 0.0
    "Fitted position angle, from North through East (degrees) at "
    "reference frequency"
    log_spec_idx: bool = True
    "True if logarithmic spectral model, False if linear. Default=True"

    source_id: Optional[str] = ""
    "ID of the source this component belongs to"
    epoch: Optional[float] = 0.0
    "Epoch of observation"

    @cached_property
    def direction(self):
        """
        Return the SkyCoord direction of the component.
        """
        return SkyCoord(ra=self.ra, dec=self.dec, unit="deg")

    def is_above_horizon(
        self, solution_time: float, array_location: EarthLocation
    ) -> bool:
        """
        Checks if the component is above horizon for the given solution time
        and array location

        Parameters
        ----------
        solution_time
            Solution time.
        array_location
            Array Locations

        Returns
        -------
            True if the given component is above the horizon at the solution
            time for the given array location.
        """
        return is_above_horizon(
            self.skycomponent,
            convert_time_to_solution_time(solution_time),
            array_location,
        )

    @cached_property
    def skycomponent(self) -> SkyComponent:
        """
        Return the equivalent
        ``ska_sdp_datamodels.sky_model.SkyComponent``.

        It is created once and cached, so it must not be modified.
        """
        return SkyComponent(
            component_id=self.component_id,
            source_id=self.source_id,
            epoch=self.epoch,
            ra_deg=self.ra,
            dec_deg=self.dec,
            i_pol_jy=self.i_pol,
            ref_freq_hz=self.ref_freq,
            a_arcsec=self.major_ax,
            b_arcsec=self.minor_ax,
            pa_deg=self.pos_ang,
            spec_idx=self.spec_idx,
            log_spec_idx=self.log_spec_idx,
        )

    def calculate_flux(self, freq: np.ndarray) -> np.ndarray:
        """
        Calculate the flux density at given frequencies.

        Computes the flux using either a power-law with a frequency-dependent
        spectral index (in log-log space) or a standard polynomial expansion
        based on the `log_spec_idx` toggle. See ``calculate_flux_at_freq``
        in ``ska_sdp_datamodels.sky_model.sky_model_operations``.

        Parameters
        ----------
        freq : np.ndarray
            The frequencies at which to evaluate the flux.

        Returns
        -------
        np.ndarray
            The calculated flux density values corresponding to the input
            frequencies.
        """
        return calculate_flux_at_freq(self.skycomponent, freq)
