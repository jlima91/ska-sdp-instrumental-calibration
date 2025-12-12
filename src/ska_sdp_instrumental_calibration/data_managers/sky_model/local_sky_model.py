import logging

import numpy as np
from astropy.coordinates.earth import EarthLocation
from astropy.coordinates.sky_coordinate import SkyCoord

from ska_sdp_instrumental_calibration.exceptions import (
    RequiredArgumentMissingException,
)

from ...numpy_processors.rotation_matrix import generate_rotation_matrices
from ..beams import BeamsFactory
from .component import Component
from .local_sky_component import LocalSkyComponent
from .sky_model_reader import generate_lsm_from_csv, generate_lsm_from_gleamegc

logger = logging.getLogger()


class LocalSkyModel:
    """
    A class representing a local sky model composed of various sky components.

    This class manages a collection of sky components and handles the
    generation of predicted visibilities based on these components for specific
    observation parameters.

    Parameters
    ----------
    components : list[Component]
        A list of component objects representing the sources in the sky model.
    solution_time : float
        The solution time associated with this sky model. This typically
        represents the timestamp or interval for which the model is valid.

    Attributes
    ----------
    components : list[Component]
        The list of sky components managed by this model.
    soln_time : float
        The specific time instance for which this model is valid.
    """

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
        """
        Calculate predicted visibilities for all components in the sky model.

        This method iterates through the model's components, calculates the
        visibility contribution of each, and sums them to produce the total
        predicted visibility for the specified baseline configuration and
        frequency channels. It optionally applies beam attenuation and Faraday
        rotation if provided.

        Parameters
        ----------
        uvw : numpy.ndarray
            The UVW coordinates of the baselines. Shape is typically
            (n_times, n_baselines, 3).
        frequency : numpy.ndarray
            The frequency channels for the simulation in Hz.
            Shape: (n_channels,).
        polarisation : numpy.ndarray
            The polarisation frames (e.g., Stokes I, Q, U, V or linear XX, XY).
            Shape: (n_pols,).
        phasecentre : SkyCoord or object
            The phase centre of the observation direction.
        antenna1 : numpy.ndarray
            Indices of the first antenna in the baseline pairs. Shape matches
            the baseline dimension of ``uvw``.
        antenna2 : numpy.ndarray
            Indices of the second antenna in the baseline pairs. Shape matches
            the baseline dimension of ``uvw``.
        beams_factory : BeamsFactory, optional
            A factory object capable of generating beam models. If provided,
            ``get_beams_low`` will be called to apply beam effects. Default is
            None.
        station_rm : numpy.ndarray, optional
            Rotation measure values for the stations to apply Faraday rotation.
            If None, no Faraday rotation is applied. Default is None.
        output_dtype : data-type, optional
            The data type of the output visibility array. Default is
            ``np.complex64``.

        Returns
        -------
        numpy.ndarray
            The total predicted visibilities summing contributions from all
            components. Shape: (n_times, n_baselines, n_channels, n_pols).

        Notes
        -----
        The method dynamically creates `LocalSkyComponent` instances for each
        component in `self.components` and aggregates their `create_vis`
        outputs.
        """
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
            skycomponent = LocalSkyComponent.create_from_component(
                comp, frequency, polarisation
            )
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
    """
    A class representing a Global Sky Model (GSM).

    This class manages the initialization of sky components from various
    catalogues (e.g., GLEAM or local CSV files) and provides functionality to
    generate local sky models for specific observation times and array
    locations.

    Parameters
    ----------
    phasecentre
        The phase centre of the observation, used as the reference point
        for the field of view search
    fov
        The field of view diameter in degrees
    flux_limit
        The minimum flux density threshold in Jy. Sources below this limit
        are excluded
    alpha0
        The default spectral index to use if not specified in the catalogue
    gleamfile
        Path to the GLEAM catalogue file. If provided, the model is
        generated from this file
    lsm_csv_path
        Path to a CSV file containing sky model components. Used if
        ``gleamfile`` is None

    Raises
    ------
    RequiredArgumentMissingException
        If neither ``gleamfile`` nor ``lsm_csv_path`` is provided.

    Attributes
    ----------
    components: list[Component]
        The list of sky components loaded into the global model
    """

    def __init__(
        self,
        phasecentre: SkyCoord,
        fov : float =10.0,
        flux_limit : float =1.0,
        alpha0 : float =-0.78,
        gleamfile : str =None,
        lsm_csv_path : str =None,
    ):
        """
        Initialize the Global Sky Model from a catalogue.

        Loads components based on a specified field of view and flux limit.
        Prioritizes the GLEAM file if both GLEAM and CSV paths are provided.
        """
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
    ) -> LocalSkyModel:
        """
        Generate a Local Sky Model for a specific time and location.

        Filters the global components to include only those currently visible
        (above the horizon) for the given array location and time.

        Parameters
        ----------
        solution_time
            The time of the observation (e.g., MJD or Unix timestamp) used to
            calculate source positions.
        array_location
            The geographical location of the telescope array.

        Returns
        -------
            A new object containing only the visible components for the
            specified parameters.
        """
        components = [
            comp
            for comp in self.components
            if comp.is_above_horizon(solution_time, array_location)
        ]
        return LocalSkyModel(components, solution_time)
