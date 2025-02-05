"""Functions for generating sky models and model visibilities"""

import logging

import everybeam as eb
import numpy as np
import xarray as xr
from astropy.coordinates import ITRS, AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from numpy import typing

logger = logging.getLogger("processing_tasks.beams")


class GenericBeams:
    """A generic class for beam handling.

    Generic interface to beams.
    Beams based on everybeam or other packages will be added or derived.

    At present for the Low array, everybeam is used to generate a common beam
    pattern for all stations
    telescope = eb.load_telescope(
        OSKAR_MOCK.ms,
        use_differential_beam=False,
        element_response_model="skala40_wave"
    )
    For other array types, all beam values are set to 2x2 identity matrices.

    Args:
        vis (xr.Dataset) dataset containing required metadata.
        array (str) array type (e.g. "low" or "mid"). By default the vis
            configuration name will be searched for an obvious match.
        direction (SkyCoord) beam direction. By default the vis phase centre
            will be used.
        ms_path (str) location of measurement set for everybeam (e.g.
            OSKAR_MOCK.ms).
    """

    def __init__(
        self,
        vis: xr.Dataset,
        array: str = None,
        direction: SkyCoord = None,
        ms_path: str = None,
    ):
        if not isinstance(vis, xr.Dataset):
            raise ValueError(f"vis is not of type xr.Dataset: {type(vis)}")

        # Can relax this, but do it like this for now...
        if vis._polarisation_frame != "linear" or len(vis.polarisation) != 4:
            raise ValueError("Beams are only defined for linear data.")

        if direction is None:
            self.beam_direction = vis.phasecentre
        else:
            self.beam_direction = direction

        # Useful metadata
        self.telescope = None
        self.antenna_names = vis.configuration.names.data
        self.array_location = vis.configuration.location
        self.antenna_locations = []
        for antenna in range(len(self.antenna_names)):
            xyz = vis.configuration.xyz.data[antenna, :]
            self.antenna_locations.append(
                EarthLocation.from_geocentric(
                    xyz[0],
                    xyz[1],
                    xyz[2],
                    unit="m",
                )
            )

        # Check beam pointing
        altaz = self.beam_direction.transform_to(
            AltAz(
                obstime=Time(vis.datetime.data[0]),
                location=self.array_location,
            )
        )
        if altaz.alt.degree < 0:
            logger.warning("pointing below horizon: %.f deg", altaz.alt.degree)

        # If array type is unset, see if it is obvious from the config
        if array is None:
            name = vis.configuration.name.lower()
            if name.find("low") >= 0:
                array = "low"
            elif name.find("mid") >= 0:
                array = "mid"
            else:
                array = ""

        # Initialise the beam models
        if array.lower() == "low":
            logger.info("Initialising beams for Low")
            self.array = array.lower()
            if ms_path is None:
                raise ValueError("Low array requires ms_path for everybeam.")
            self.telescope = eb.load_telescope(
                ms_path,
                use_differential_beam=False,
                element_response_model="skala40_wave",
            )
            self.delay_dir_itrf = None
            self.normalise = np.zeros((len(vis.frequency), 2, 2), "complex")
            self.normalise[..., :, :] = np.eye(2)
        elif array.lower() == "mid":
            logger.info("Initialising beams for Mid")
            self.array = array.lower()
            logger.warning(
                "The Mid beam model is not current set. "
                "Only use with compact, centred sky models."
            )
        else:
            logger.info("Unknown beam")

    def update_beam_direction(self, direction: SkyCoord):
        """Return the response of each antenna or station in a given direction.

        :param direction: Pointing direction of the beams
        """
        self.beam_direction = direction

    def update_beam(self, frequency: typing.NDArray[float], time: Time):
        """Update the ITRF coordinates of the beam and normalisation factors.

        :param frequency: 1D array of frequencies
        :param time: obstime
        """
        station_id = 0
        self.delay_dir_itrf = radec_to_xyz(self.beam_direction, time)
        for chan, freq in enumerate(frequency):
            self.normalise[chan] = np.linalg.inv(
                # This is normalising in be beam dir, but should be zenith
                self.telescope.station_response(
                    time.mjd * 86400,
                    station_id,
                    freq,
                    self.delay_dir_itrf,
                    self.delay_dir_itrf,
                )
            )

    def array_response(
        self,
        direction: SkyCoord,
        frequency: typing.NDArray[float],
        time: Time = None,
    ) -> typing.NDArray[complex]:
        """Return the response of each antenna or station in a given direction

        :param direction: Direction of desired response
        :param frequency: 1D array of frequencies
        :param time: obstime
        :return: array of beam matrices [nant, nfreq, 2, 2]
        """
        if time is not None:
            altaz = direction.transform_to(
                AltAz(obstime=time, location=self.array_location)
            )
            if altaz.alt.degree < 0:
                logger.warning(
                    "Direction below horizon. Returning zero gains."
                )
                return np.zeros(
                    (len(self.antenna_names), len(frequency), 2, 2), "complex"
                )

        beams = np.empty(
            (len(self.antenna_names), len(frequency), 2, 2), "complex"
        )
        if self.array == "low":
            if time is None:
                raise ValueError("Time must be specified for the Low beam.")

            # Get the station pointing direction in ITRF
            if self.delay_dir_itrf is None:
                self.delay_dir_itrf = radec_to_xyz(self.beam_direction, time)

            # Get the component direction in ITRF
            dir_itrf = radec_to_xyz(direction, time)

            mjds = time.mjd * 86400

            for stn_id in range(len(self.antenna_names)):
                for chan, freq in enumerate(frequency):
                    beams[stn_id, chan, :, :] = (
                        self.telescope.station_response(
                            mjds, stn_id, freq, dir_itrf, self.delay_dir_itrf
                        )
                        @ self.normalise[chan]
                    )
            # np.set_printoptions(linewidth=120, precision=4, suppress=True)
            # print(
            #     f"sep = {direction.separation(self.beam_direction):.1f}, "
            #     + f"response = {beams[0, 0, :, :].reshape(4)}"
            # )

        else:
            beams[..., :, :] = np.eye(2)

        return beams


# from everybeam.readthedocs.io/en/latest/tree/demos/lofar-array-factor.html
def radec_to_xyz(dir_pointing: SkyCoord, time: Time):
    """
    Convert RA and Dec ICRS coordinates to ITRS cartesian coordinates.

    Args:
        dir_pointing (SkyCoord): astropy pointing direction
        time (Time): astropy obstime

    :param dir_pointing: SkyCoord direction in ICRS ra, dec coordinates
    :param time: astropy obstime
    :return: NumPy array containing the ITRS X, Y and Z coordinates
    """
    dir_pointing_itrs = dir_pointing.transform_to(ITRS(obstime=time))
    return np.asarray(dir_pointing_itrs.cartesian.xyz.transpose())
