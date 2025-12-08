"""Functions for generating sky models and model visibilities"""

import logging

import everybeam as eb
import numpy as np
import numpy.typing as npt
import xarray as xr
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from ..data_managers.beams import radec_to_xyz

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

    Parameters
    ----------
    vis: xr.Dataset
        Dataset containing required metadata.
    array: str
        array type (e.g. "low" or "mid"). By default the vis
        configuration name will be searched for an obvious match.
    direction: SkyCoord
        Beam direction. By default the vis phase centre
        will be used.
    ms_path: str
        Location of measurement set for everybeam (e.g. OSKAR_MOCK.ms).
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
            self.telescope = eb.load_telescope(  # pylint: disable=I1101
                ms_path
            )
            self.delay_dir_itrf = None
            self.set_scale = None
            if type(self.telescope) is eb.OSKAR:  # pylint: disable=I1101
                # why not just set the normalisation now?
                logger.info("Setting beam normalisation for OSKAR data")
                self.set_scale = "oskar"
            self.scale = np.ones(len(vis.frequency))
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

    def update_beam(self, frequency: npt.NDArray[float], time: Time):
        """Update the ITRF coordinates of the beam and normalisation factors.

        :param frequency: 1D array of frequencies
        :param time: obstime
        """
        # Coordinates of beam centre
        self.delay_dir_itrf = radec_to_xyz(self.beam_direction, time)

        if self.set_scale is None:
            # Normalisation scale is already set
            pass

        elif self.set_scale == "oskar":
            # Set normalisation scaling to the Frobenius norm of the zenith
            # response divided by sqrt(2).
            # Should be the same for all stations so pick one. Should use the
            # station location rather than central array location, e.g. using
            # the following code, but some functions (e.g. ska-sdp-datamodels
            # function create_named_configuration -- at least for some
            # configurations) set xyz coordinates to ENU rather than the
            # geocentric coordinates. So use the array location and a central
            # station for now. Note that OSKAR datasets have correct geocentric
            # coordinates, but also have the array location set to the first
            # station xyz, so using array_location with stn=0 works.
            #     xyz = vis.configuration.xyz.data[stn, :]
            #     self.antenna_locations.append(
            #         EarthLocation.from_geocentric(
            #             xyz[0], xyz[1], xyz[2], unit="m",
            #         )
            #     )
            stn = 0
            dir_itrf_zen = radec_to_xyz(
                SkyCoord(
                    alt=90,
                    az=0,
                    unit="deg",
                    frame="altaz",
                    obstime=time,
                    location=self.array_location,
                ),
                time,
            )
            for chan, freq in enumerate(frequency):
                J = self.telescope.station_response(
                    time.mjd * 86400,
                    stn,
                    freq,
                    dir_itrf_zen,
                    dir_itrf_zen,
                )
                self.scale[chan] = np.sqrt(2) / np.linalg.norm(J)

            # only need to do this once, so set to None when finished
            self.set_scale = None

        else:
            raise ValueError("Unknown beam normalisation.")

    def array_response(
        self,
        direction: SkyCoord,
        frequency: npt.NDArray[float],
        time: Time = None,
    ) -> npt.NDArray[complex]:
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

            for stn in range(len(self.antenna_names)):
                for chan, freq in enumerate(frequency):
                    beams[stn, chan, :, :] = (
                        self.telescope.station_response(
                            mjds, stn, freq, dir_itrf, self.delay_dir_itrf
                        )
                        * self.scale[chan]
                    )

        else:
            beams[..., :, :] = np.eye(2)

        return beams
