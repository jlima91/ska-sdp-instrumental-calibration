import logging
import os

import dask.array as da
import everybeam as eb
import numpy as np
import xarray as xr
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from ska_sdp_datamodels.configuration import Configuration

from ska_sdp_instrumental_calibration.processing_tasks.beams import (
    radec_to_xyz,
)

logger = logging.getLogger()


def station_response_beam_ufunc(
    frequency,
    scale,
    nstations,
    beam_ms,
    mjds,
    delay_dir_itrf,
    dir_itrf,
    beams_dtype,
):
    beams = np.empty((len(frequency), nstations, 2, 2), dtype=beams_dtype)

    # TODO: Fix this nested looping
    telescope = eb.load_telescope(beam_ms)
    for chan, freq in enumerate(frequency):
        for stn in range(nstations):
            beams[chan, stn, :, :] = (
                telescope.station_response(
                    mjds, stn, freq, dir_itrf, delay_dir_itrf
                )
                * scale[chan]
            )

    return beams


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
        configuration: Configuration,
        direction: SkyCoord,
        array: str = None,
        ms_path: str = None,
    ):
        self.beam_direction = direction
        self.beam_ms = ms_path
        self.telescope = None

        # Useful metadata
        self.antenna_names = configuration.names.data
        self.array_location = configuration.location

        # TODO: Is this check necessary? This requires entire
        # datetime array to be in memory, so needs compute
        # Check beam pointing
        # altaz = self.beam_direction.transform_to(
        #     AltAz(
        #         obstime=Time(datetime.data[0]),
        #         location=self.array_location,
        #     )
        # )
        # if altaz.alt.degree < 0:
        #     logger.warning(
        #         "pointing below horizon: %.f deg",
        #         altaz.alt.degree)

        # If array type is unset, see if it is obvious from the config
        if array is None:
            name = configuration.name.lower()
            if name.find("low") >= 0:
                array = "low"
            elif name.find("mid") >= 0:
                array = "mid"
            else:
                array = ""

        self.scale = None
        self.set_scale = None
        self.delay_dir_itrf = None
        # Initialise the beam models
        if array.lower() == "low":
            logger.info("Initialising beams for Low")
            self.array = array.lower()
            if self.beam_ms is None:
                raise ValueError("Low array requires ms_path for everybeam.")
            self.telescope = eb.load_telescope(self.beam_ms)
            if type(self.telescope) is eb.OSKAR:
                # why not just set the normalisation now?
                logger.info("Setting beam normalisation for OSKAR data")
                self.set_scale = "oskar"
            else:
                self.set_scale = "ones"

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

    def update_beam(self, frequency_xdr: xr.DataArray, time: Time):
        """Update the ITRF coordinates of the beam and normalisation factors.
        Sets the "scale" attribute to a chunked xarray Dataarray
        with frequency coordinates.

        :param frequency: xr.Datarray of dask chuncked frequency values
        :param time: obstime
        """
        # Coordinates of beam centre
        self.delay_dir_itrf = radec_to_xyz(self.beam_direction, time)

        if self.set_scale is None:
            # Normalisation scale is already set
            pass

        elif self.set_scale == "ones":
            self.scale = xr.ones_like(frequency_xdr)
            # only need to do this once, so set to None when finished
            self.set_scale = None

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

            # TODO: Dask delay this if needed.
            # Using coords['frequency'] as that is a in-mem numpy array
            # while the frequency_xdr.data is dask array
            new_scale = np.ones_like(frequency_xdr.coords["frequency"])
            for chan, freq in enumerate(frequency_xdr.coords["frequency"]):
                J = self.telescope.station_response(
                    time.mjd * 86400,
                    stn,
                    freq,
                    dir_itrf_zen,
                    dir_itrf_zen,
                )
                new_scale[chan] = np.sqrt(2) / np.linalg.norm(J)

            self.scale = xr.ones_like(frequency_xdr) * new_scale
            # only need to do this once, so set to None when finished
            self.set_scale = None

        else:
            raise ValueError("Unknown beam normalisation.")

    def array_response(
        self,
        direction: SkyCoord,
        frequency_xdr: xr.DataArray,
        time: Time = None,
        output_dtype=np.complex128,
    ) -> xr.DataArray:
        """Return the response of each antenna or station in a given direction

        :param direction: Direction of desired response
        :param frequency: 1D array of frequencies
        :param time: obstime
        :return: array of beam matrices [nant, nfreq, 2, 2]
        """
        nstations = len(self.antenna_names)

        if time is not None:
            altaz = direction.transform_to(
                AltAz(obstime=time, location=self.array_location)
            )
            if altaz.alt.degree < 0:
                logger.warning(
                    "Direction below horizon. Returning zero gains."
                )
                return xr.DataArray(
                    da.zeros(
                        (frequency_xdr.size, nstations, 2, 2),
                        dtype=output_dtype,
                    ),
                    dims=[
                        "frequency",
                        "id",
                        "x",
                        "y",
                    ],
                    coords={"frequency": frequency_xdr.coords["frequency"]},
                )

        if self.array != "low":
            logger.info(
                "Currently only 'low' array is supported"
                "Returning identity matrix"
            )
            return xr.DataArray(
                da.broadcast_to(
                    da.eye(2, dtype=output_dtype),
                    shape=(frequency_xdr.size, nstations, 2, 2),
                ),
                dims=[
                    "frequency",
                    "id",
                    "x",
                    "y",
                ],
                coords={"frequency": frequency_xdr.coords["frequency"]},
            )

        # Processing for 'low' array
        if time is None:
            raise ValueError("Time must be specified for the Low beam.")

        if self.scale is None:
            raise AttributeError(
                "Attribute 'scale' for beams is not set. "
                "Please call 'update_beam' to set scale attribute."
            )

        # Get the station pointing direction in ITRF
        if self.delay_dir_itrf is None:
            self.delay_dir_itrf = radec_to_xyz(self.beam_direction, time)

        # Get the component direction in ITRF
        dir_itrf = radec_to_xyz(direction, time)

        mjds = time.mjd * 86400

        return xr.apply_ufunc(
            station_response_beam_ufunc,
            frequency_xdr,
            self.scale,
            input_core_dims=[[], []],
            output_core_dims=[
                (
                    "id",
                    "x",
                    "y",
                ),
            ],
            dask_gufunc_kwargs={
                "output_sizes": {"id": nstations, "x": 2, "y": 2},
            },
            dask="parallelized",
            output_dtypes=[output_dtype],
            kwargs={
                "nstations": nstations,
                "beam_ms": self.beam_ms,
                "mjds": mjds,
                "delay_dir_itrf": self.delay_dir_itrf,
                "dir_itrf": dir_itrf,
                "beams_dtype": output_dtype,
            },
        )


def create_beams(
    time: Time,
    frequency: xr.DataArray,
    configuration: Configuration,
    phasecentre: SkyCoord,
    eb_coeffs: str,
    eb_ms: str,
):
    # Could do this once externally, but don't want to pass around
    # exotic data types.
    os.environ["EVERYBEAM_DATADIR"] = eb_coeffs

    beams = GenericBeams(
        configuration=configuration,
        direction=phasecentre,
        array="low",
        ms_path=eb_ms,
    )

    # Update ITRF coordinates of the beam and normalisation factors
    beams.update_beam(frequency, time=time)

    # Check beam pointing direction
    altaz = beams.beam_direction.transform_to(
        AltAz(obstime=time, location=beams.array_location)
    )
    if altaz.alt.degree < 0:
        raise ValueError(f"Pointing below horizon el={altaz.alt.degree}")

    return beams
