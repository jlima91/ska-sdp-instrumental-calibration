import logging
from dataclasses import dataclass

import numpy as np
from astropy.coordinates import ITRS, AltAz, SkyCoord
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time

from .telescope import Telescope

logger = logging.getLogger(__name__)


def convert_time_to_solution_time(time: float | np.ndarray) -> Time:
    """
    Convert float time to Astropy time.

    Parameters
    ----------
    time: float
        Input time in float

    Return
    ------
      Time object containing an array of time values
    """
    return Time(time / 86400.0, format="mjd", scale="utc")


# from everybeam.readthedocs.io/en/latest/tree/demos/lofar-array-factor.html
def radec_to_xyz(direction: SkyCoord, time: Time):
    """
    Convert RA and Dec ICRS coordinates to ITRS cartesian coordinates.

    Parameters
    ----------
    direction: SkyCoord
        astropy pointing direction
    time: astropy.time.Timne
        Observation time

    Return
    ------
    NumPy array containing the ITRS X, Y and Z coordinates
    """
    direction_itrs = direction.transform_to(ITRS(obstime=time))
    return np.asarray(direction_itrs.cartesian.xyz.transpose())


class PointingBelowHorizon(Exception):
    """
    Pointing below exception raised when a sky component with its RA-DEC at a
    given time is below horizon
    """

    pass


class BeamsLow:
    """A beam class specific to handling low beams."""

    def __init__(
        self,
        array_location: EarthLocation,
        direction: SkyCoord,
        frequency: np.ndarray,
        soln_time: float,
        telescope: Telescope,
    ):

        self.array_location = array_location
        self.beam_direction = direction
        self.frequency = frequency
        self.telescope = telescope

        self.delay_dir_itrf = None

        self.solution_time = convert_time_to_solution_time(soln_time)

        self.solution_time_mjd_seconds = soln_time

        # Check beam pointing direction for all solution times
        self.validate_direction_above_horizon(self.beam_direction)

        # Coordinates of beam centre
        self.delay_dir_itrf = radec_to_xyz(
            self.beam_direction, self.solution_time
        )

    def validate_direction_above_horizon(self, direction: SkyCoord) -> AltAz:
        """
        Calculate Altitude-Azimuth for a direction, and ensure that the
        direction is valid for the given solution interval of the beam

        Parameters
        ----------
        direction: SkyCoord
           direction of the source

        Raises
        ------
        PointingBelowHorizon exception if direction is below horizon

        Returns
        -------
        Altaz[list] if direction is valid
        """
        altaz = direction.transform_to(
            AltAz(obstime=self.solution_time, location=self.array_location)
        )
        if altaz.alt.degree < 0:
            raise PointingBelowHorizon(
                "Pointing below horizon for some of the solution times"
            )

    def array_response(
        self,
        direction: SkyCoord,
    ) -> np.ndarray:
        """Return the response of each antenna or station in a given direction

        Parameters
        ----------
        direction: SkyCoord
            Direction of desired response

        Returns
        -------
        np.complex128 array of beam matrices [nant, nfreq, 2, 2]
        """
        # Get the component direction in ITRF
        dir_itrf = radec_to_xyz(direction, self.solution_time)

        return self.telescope.station_response(
            solution_time=self.solution_time_mjd_seconds,
            frequencies=self.frequency,
            direction=dir_itrf,
            station0=self.delay_dir_itrf,
        ).squeeze(axis=0)


@dataclass
class BeamsFactory:
    array_location: EarthLocation
    direction: SkyCoord
    telescope: Telescope

    def get_beams_low(self, frequency, soln_time) -> BeamsLow:
        """
        Initializes and returns a BeamsLow object.

        Parameters
        ----------
        frequency: np.ndarray
            Array of frequencies
        soln_time: float
            Solution time for the given solution interval

        Returns
        -------
            A BeamsLow Object for the given frequency range and solution time.
        """
        return BeamsLow(
            array_location=self.array_location,
            direction=self.direction,
            frequency=frequency,
            soln_time=soln_time,
            telescope=self.telescope,
        )
