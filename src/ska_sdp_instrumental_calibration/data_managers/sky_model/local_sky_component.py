import numpy as np
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.calibration import apply_antenna_gains_to_visibility
from ska_sdp_func_python.imaging import dft_skycomponent

from ..beams import BeamsLow
from .component import Component


class LocalSkyComponent(SkyComponent):

    @staticmethod
    def create_from_component(comp: Component, freq: np.ndarray, *_):
        """
        Construct a LocalSkyComponent from a Component.

        All sources are unpolarised and specified in the linear polarisation
        frame using XX = YY = Stokes I.

        After deconvolving the gaussian for a given component, components
        with non-zero widths after this process are stored with
        shape = "GAUSSIAN". Otherwise shape = "POINT".

        Parameters
        ----------
        comp : Component
            An instance of Component
        freq : ndarray
            An array of frequency values in Hz

        Returns
        -------
        LocalSkyComponent
            An instance of LocalSkyComponent
        """
        freq = np.array(freq)

        flux0 = comp.flux
        freq0 = comp.ref_freq
        alpha = comp.alpha

        # assume 4 pols
        flux = np.zeros((len(freq), 4))

        # Convention as disscussed in "2025-09-23 Low G3 SDP Sync"
        # meeting on 23 Sept 2025 08:00 - 09:00 UTC.
        flux[:, 0] = flux[:, 3] = flux0 * np.power((freq / freq0), alpha)

        # Deconvolve synthesised beam from fitted shape parameters.
        smaj, smin, spa = comp.deconvolve_gaussian()
        if smaj == 0 and smin == 0:
            shape = "POINT"
            params = {}
        else:
            shape = "GAUSSIAN"
            # From what I can tell, all params units are degrees
            params = {
                "bmaj": smaj / 3600.0,
                "bmin": smin / 3600.0,
                "bpa": spa,
            }

        return LocalSkyComponent(
            direction=comp.direction,
            frequency=freq,
            name=comp.name,
            flux=flux,
            polarisation_frame=PolarisationFrame("linear"),
            shape=shape,
            params=params,
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
            UVW coordinates of shape (time, frequency, baselineid, spatial).
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
            4D faraday rotation matrix of shape (antenna, frequency, 2, 2).

        Returns
        -------
        np.ndarray
            Visibility for the given LocalSkyComponent. This has shape
            (time, frequency, baselineid, polarisation).
        """
        sky_comp_vis = dft_skycomponent(
            uvw=uvw, skycomponent=self, phase_centre=phasecentre
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
