"""
LocalSkyComponent backed by the
``ska_sdp_datamodels.global_sky_model.SkyComponent`` dataclass.

It evaluates a sky component at the frequencies of an observation and
predicts its visibilities.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.global_sky_model import SkyComponent
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_func_python.calibration import apply_antenna_gains_to_visibility

from ..beams import BeamsLow
from .sky_model_operations import dft_skycomponent


@dataclass
class LocalSkyComponent:
    """
    A sky component evaluated at the frequencies of an observation.

    All sources are unpolarised and specified in the linear polarisation
    frame using XX = YY = Stokes I.
    """

    skycomponent: SkyComponent
    "Sky component from the global / local sky model"
    frequency: np.ndarray
    "Frequencies of shape (n_freqs,). Units: [Hz]"
    polarisation_frame: PolarisationFrame = field(
        default_factory=lambda: PolarisationFrame("linear")
    )
    "Polarisation frame of the predicted visibilities"

    @staticmethod
    def create_from_component(
        comp: SkyComponent, freq: np.ndarray, *_
    ) -> "LocalSkyComponent":
        """
        Construct a LocalSkyComponent from a SkyComponent.

        Convention as discussed in "2025-09-23 Low G3 SDP Sync"
        meeting on 23 Sept 2025 08:00 - 09:00 UTC: all sources are
        unpolarised and specified in the linear polarisation frame using
        XX = YY = Stokes I.

        Parameters
        ----------
        comp : SkyComponent
            An instance of SkyComponent. Its Gaussian shape parameters
            are expected to be already deconvolved.
        freq : ndarray
            Frequencies of shape (n_freqs,). Units: [Hz]

        Returns
        -------
        LocalSkyComponent
            An instance of LocalSkyComponent
        """
        return LocalSkyComponent(
            skycomponent=comp,
            frequency=np.array(freq),
            polarisation_frame=PolarisationFrame("linear"),
        )

    @cached_property
    def direction(self) -> SkyCoord:
        """
        Return the SkyCoord direction of the component, built from its
        ``ra_deg`` and ``dec_deg``.
        """
        return SkyCoord(
            ra=self.skycomponent.ra_deg,
            dec=self.skycomponent.dec_deg,
            unit="deg",
        )

    def create_vis(
        self,
        uvw,
        phasecentre,
        antenna1,
        antenna2,
        beams: BeamsLow = None,
        faraday_rot_matrix: np.ndarray = None,
    ):
        """
        Create visibility for a given LocalSkyComponent.

        Parameters
        ----------
        uvw : np.ndarray
            UVW coordinates of shape (time, baselineid, spatial).
            Units: [m]
        phasecentre : SkyCoord
            Phase centre of the observation.
        antenna1 : np.ndarray
            The indices of 1st antenna in each pair of baseline. Must be of
            shape (baselineid,).
        antenna2 : np.ndarray
            The indices of 2nd antenna in each pair of baseline. Must be of
            shape (baselineid,).
        beams : BeamsLow, optional
            Beams object containing the primary beam.
        faraday_rot_matrix : np.ndarray, optional
            Faraday rotation matrix of shape (1, antenna, frequency, 2, 2),
            where the first axis is a time axis of length 1.
            Dimensionless.

        Returns
        -------
        np.ndarray
            Visibility for the given LocalSkyComponent. This has shape
            (time, baselineid, frequency, polarisation). Units: [Jy]
        """
        sky_comp_vis = dft_skycomponent(
            uvw=uvw,
            skycomponent=self.skycomponent,
            phase_centre=phasecentre,
            frequency=self.frequency,
            polarisation_frame=self.polarisation_frame,
        )
        if beams is None and faraday_rot_matrix is None:
            return sky_comp_vis

        component_array_response = faraday_rot_matrix
        if beams is not None:
            component_array_response = beams.array_response(
                direction=self.direction
            )[np.newaxis, ...]

            if faraday_rot_matrix is not None:
                component_array_response = (
                    component_array_response @ faraday_rot_matrix
                )

        return apply_antenna_gains_to_visibility(
            sky_comp_vis,
            component_array_response,
            antenna1,
            antenna2,
        )
