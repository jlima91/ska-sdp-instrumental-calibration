import logging

import numpy as np
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates.sky_coordinate import SkyCoord

from ska_sdp_instrumental_calibration.exceptions import (
    RequiredArgumentMissingException,
)

from ..numpy_processors.lsm import (
    Component,
    generate_lsm_from_csv,
    generate_lsm_from_gleamegc,
)
from ..numpy_processors.rotation_matrix import generate_rotation_matrices
from .beams import BeamsFactory

logger = logging.getLogger()


class LocalSkyModel:
    def __init__(self, components: list[Component], solution_time: float):
        self.components = components
        self.soln_time = solution_time

    def create_vis(
        self,
        uvw,
        frequency,
        polarisation,
        phasecentre,
        antenna1,
        antenna2,
        beams_factory: BeamsFactory = None,
        station_rm=None,
        output_dtype=np.complex64,
    ):
        faraday_rot_matrix = None
        if station_rm is not None:
            faraday_rot_matrix = generate_rotation_matrices(
                station_rm, frequency, output_dtype
            )[
                np.newaxis, ...
            ]  # Add time axis at the start

        beams = None
        if beams_factory is not None:
            beams = beams_factory.get_beams_low(frequency, self.soln_time)

        predicted_vis = np.zeros(
            (*uvw.shape[:2], frequency.size, polarisation.size),
            dtype=output_dtype,
        )
        for comp in self.components:
            skycomponent = comp.get_skycomponent(frequency, polarisation)
            predicted_vis = predicted_vis + skycomponent.create_vis(
                uvw,
                phasecentre,
                antenna1,
                antenna2,
                beams,
                faraday_rot_matrix,
            )

        return predicted_vis


class GlobalSkyModel:
    components: list[Component]

    def __init__(
        self,
        phasecentre: SkyCoord,
        fov=10.0,
        flux_limit=1.0,
        alpha0=-0.78,
        gleamfile=None,
        lsm_csv_path=None,
    ):
        if gleamfile is not None and lsm_csv_path is not None:
            logger.warning("GSM: GLEAMFILE and CSV provided. Using GLEAMFILE")

        logger.info("Generating GSM for predict with:")
        logger.info(f" - Search radius: {fov/2} deg")
        logger.info(f" - Flux limit: {flux_limit} Jy")

        if gleamfile is not None:
            logger.info(f" - Catalogue file: {gleamfile}")
            lsm = generate_lsm_from_gleamegc(
                gleamfile=gleamfile,
                phasecentre=phasecentre,
                fov=fov,
                flux_limit=flux_limit,
                alpha0=alpha0,
            )
        elif lsm_csv_path is not None:
            logger.info(f" - Catalogue file: {lsm_csv_path}")
            lsm = generate_lsm_from_csv(
                csvfile=lsm_csv_path,
                phasecentre=phasecentre,
                fov=fov,
                flux_limit=flux_limit,
            )
        else:
            raise RequiredArgumentMissingException(
                "No GSM components provided. "
                "Either provide GLEAMFILE or GSM CSV file"
            )

        logger.info(f"GSM: found {len(lsm)} components")

        self.components = lsm

    def get_local_sky_model(
        self, solution_time: float, array_location: EarthLocation
    ):
        components = [
            comp
            for comp in self.components
            if comp.is_above_horizon(solution_time, array_location)
        ]
        return LocalSkyModel(components, solution_time)
