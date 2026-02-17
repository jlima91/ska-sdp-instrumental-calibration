"""Module for generating the local sky model.

Note that these are temporary functions that will be replaced by functions that
connect to ska-sdp-global-sky-model functions.
"""

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord

from ..beams import convert_time_to_solution_time
from .flux_utils import calculate_flux_for_spectral_indices

logger = logging.getLogger(__name__)


@dataclass
class Component:
    """
    Class for LSM components.

    Class to hold catalogue data for a component of the local sky model. For
    components with elliptical Gaussian parameters, if beam parameters are also
    supplied, the beam will be deconvolved from the component parameters. If
    the component parameters have already had the beam deconvolved, the beam
    parameters should be left at zero.
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
    "Fitted semi-major axis (arcsec) at reference frequency"
    minor_ax: Optional[float] = 0.0
    "Fitted semi-minor axis (arcsec) at reference frequency"
    pos_ang: Optional[float] = 0.0
    "Fitted position angle (degrees) at reference frequency"
    beam_major: float = 0.0
    "Semi-major axis of a beam that is still convolved into the "
    "main component shape (arcsec). Default=0 (no beam present)"
    beam_minor: float = 0.0
    "Semi-minor axis of a beam that is still convolved into the "
    "main component shape (arcsec). Default=0 (no beam present)"
    beam_pa: float = 0.0
    "Position angle of a beam that is still convolved into the "
    "main component shape (degrees). Default=0"
    log_spec_idx: bool = True
    "True if logarithmic spectral model, False if linear. Default=True"

    @cached_property
    def direction(self):
        """
        Return the SkyCoord direction of the component.
        """
        return SkyCoord(ra=self.ra, dec=self.dec, unit="deg")

    def get_altaz(
        self, solution_time: float, array_location: EarthLocation
    ) -> SkyCoord:
        """
        Get the AltAz coordinate of the component at given solution time
        and array location.

        Parameters
        ----------
        solution_time
            Solution time (seconds since mjd epoch)
        array_location
            Location of the array.

        Returns
        -------
            A new object with the component's direction coordinate
            represented in the given AltAz frame.

        Notes
        -----
        The solution time is converted to a datetime object using the
        py:func:`convert_time_to_solution_time` function before being passed to
        `AltAz`.
        """
        return self.direction.transform_to(
            AltAz(
                obstime=convert_time_to_solution_time(solution_time),
                location=array_location,
            )
        )

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
        return self.get_altaz(solution_time, array_location).alt.degree >= 0

    def deconvolve_gaussian(self) -> tuple[float, float, float]:
        """
        Deconvolve MWA synthesised beam from Gaussian shape parameters.

        This follows the approach of the analysisutilities function
        deconvolveGaussian in the askap-analysis repository, written by Matthew
        Whiting. This is based on the approach described in Wild (1970),
        AuJPh 23, 113.

        Returns
        -------
            Tuple of deconvolved parameters (same units as data in self)
        """
        if (
            self.major_ax is None
            or self.minor_ax is None
            or self.pos_ang is None
        ):
            return 0.0, 0.0, 90.0

        # fitted data on source
        fmajsq = self.major_ax * self.major_ax
        fminsq = self.minor_ax * self.minor_ax
        fdiff = fmajsq - fminsq
        fphi = 2.0 * self.pos_ang * np.pi / 180.0

        # beam data at source location
        bmajsq = self.beam_major * self.beam_major
        bminsq = self.beam_minor * self.beam_minor
        bdiff = bmajsq - bminsq
        bphi = 2.0 * self.beam_pa * np.pi / 180.0

        # source data after deconvolution
        if fdiff < 1e-6:
            # Circular Gaussian case
            smaj = np.sqrt(fmajsq - bminsq)
            smin = np.sqrt(fmajsq - bmajsq)
            psmaj = np.pi / 2.0 + self.beam_pa * np.pi / 180.0

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

    def calculate_flux(self, freq: np.ndarray) -> np.ndarray:
        spec_idx = self.spec_idx
        if spec_idx is None or spec_idx == []:
            spec_idx = [0.0]

        return calculate_flux_for_spectral_indices(
            flux=self.i_pol,
            freq=freq,
            ref_freq=self.ref_freq,
            spec_idx=spec_idx,
            log_spec_idx=self.log_spec_idx,
        )
