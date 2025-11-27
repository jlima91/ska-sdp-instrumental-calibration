import numpy as np
from ska_sdp_datamodels.sky_model import SkyComponent

from ..data_managers.gaintable import apply_antenna_gains_to_visibility
from ..processing_tasks.dft import dft_skycomponent
from ..processing_tasks.lsm.beams import BeamsLow


class LocalSkyComponent(SkyComponent):

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
