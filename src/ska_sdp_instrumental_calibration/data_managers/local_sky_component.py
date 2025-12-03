import numpy as np
from numpy import typing
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent

from ..numpy_processors.dft import dft_skycomponent
from .beams import BeamsLow
from .component import Component
from .gaintable import apply_antenna_gains_to_visibility


class LocalSkyComponent(SkyComponent):

    @staticmethod
    def create_from_component(
        comp: Component, freq: typing.NDArray[float], _=None
    ):
        """Convert the LocalSkyModel to a list of SkyComponents.

        All sources are unpolarised and specified in the linear polarisation
        frame using XX = YY = Stokes I.

        Function :func:`~deconvolve_gaussian` is used to deconvolve the MWA
        synthesised beam from catalogue shape parameters of each component.
        Components with non-zero widths after this process are stored with
        shape = "GAUSSIAN". Otherwise shape = "POINT".

        :param model: Component list
        :param freq: Frequency list in Hz
        :param freq0: Reference Frequency for flux scaling in Hz. Default is
        200e6.
            Note: freq0 should really be part of the sky model
        :return: SkyComponent list
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
