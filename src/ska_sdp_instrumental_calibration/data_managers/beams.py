import logging
from dataclasses import dataclass

import everybeam as eb
import numpy as np
from astropy.coordinates import ITRS, AltAz, SkyCoord
from astropy.coordinates.earth import EarthLocation
from astropy.time import Time

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
    # This Time->Time conversion is what was originally done in INST
    # Although Time can be treated as a "Value Object", there are subtle
    # differences in the order in which floating point operations are performed
    # thus the results may slightly vary if you try to simplify this
    # By removeing the additional datetime64 to time conversion
    return Time(Time(time / 86400.0, format="mjd", scale="utc").datetime64)


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
        nstations: int,
        array_location: EarthLocation,
        direction: SkyCoord,
        frequency: np.ndarray,
        ms_path: str,
        soln_time: float,
    ):
        self.nstations = nstations
        self.array_location = array_location
        self.beam_direction = direction
        self.frequency = frequency
        self.beam_ms = ms_path

        self.delay_dir_itrf = None

        self.solution_time = convert_time_to_solution_time(soln_time)

        self.solution_time_mjd_seconds = self.solution_time.mjd * 86400

        # Check beam pointing direction for all solution times
        self.validate_direction_above_horizon(self.beam_direction)

        # Coordinates of beam centre
        self.delay_dir_itrf = radec_to_xyz(
            self.beam_direction, self.solution_time
        )

        self.telescope = eb.load_telescope(  # pylint: disable=I1101
            self.beam_ms
        )

        self.scale = np.ones(
            (self.frequency.size,),
            dtype=self.frequency.dtype,
        )
        if type(self.telescope) is eb.OSKAR:  # pylint: disable=I1101
            """
            Set normalisation scaling to the Frobenius norm of the zenith
            response divided by sqrt(2).
            Should be the same for all stations so pick one. Should use the
            station location rather than central array location, e.g. using
            the following code, but some functions (e.g. ska-sdp-datamodels
            function create_named_configuration -- at least for some
            configurations) set xyz coordinates to ENU rather than the
            geocentric coordinates. So use the array location and a central
            station for now. Note that OSKAR datasets have correct geocentric
            coordinates, but also have the array location set to the first
            station xyz, so using array_location with stn=0 works.
                xyz = vis.configuration.xyz.data[stn, :]
                self.antenna_locations.append(
                    EarthLocation.from_geocentric(
                        xyz[0], xyz[1], xyz[2], unit="m",
                    )
                )
            """
            logger.info("Setting beam normalisation for OSKAR data")

            stn = 0

            dir_itrf_zen = radec_to_xyz(
                SkyCoord(
                    alt=90,
                    az=0,
                    unit="deg",
                    frame="altaz",
                    obstime=self.solution_time,
                    location=self.array_location,
                ),
                self.solution_time,
            )

            for chan, freq in enumerate(self.frequency):
                J = self.telescope.station_response(
                    self.solution_time_mjd_seconds,
                    stn,
                    freq,
                    dir_itrf_zen,
                    dir_itrf_zen,
                )
                self.scale[chan] = np.sqrt(2) / np.linalg.norm(J)

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

        beams = np.empty(
            (
                self.nstations,
                self.frequency.size,
                2,
                2,
            ),
            dtype=np.complex128,
        )

        # NOTE: Check if station names can be used instead of
        # station indices
        for stn in range(self.nstations):
            for chan, freq in enumerate(self.frequency):
                beams[stn, chan, :, :] = (
                    self.telescope.station_response(
                        self.solution_time_mjd_seconds,
                        stn,
                        freq,
                        dir_itrf,
                        self.delay_dir_itrf,
                    )
                    * self.scale[chan]
                )

        return beams


@dataclass
class BeamsFactory:
    """
    Dataclass to denote a beam.
    """

    nstations: int
    array_location: EarthLocation
    direction: SkyCoord
    ms_path: str

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
            **self.__dict__, frequency=frequency, soln_time=soln_time
        )
