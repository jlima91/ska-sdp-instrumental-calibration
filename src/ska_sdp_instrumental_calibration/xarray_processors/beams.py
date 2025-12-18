import logging

import numpy as np
import xarray as xr
from ska_sdp_datamodels.calibration import GainTable

from ..data_managers.beams import BeamsFactory
from ._utils import with_chunks

logger = logging.getLogger(__name__)


def _prediction_central_beams_ufunc(
    frequency: np.ndarray,
    soln_time: float,
    beams_factory: BeamsFactory,
):
    """
    frequency: np.ndarray, (frequency,)
    soln_time: float
    beams_factory: BeamsFactory

    Returns
    -------
    np.ndarray (frequency, antenna, nrec1, nrec2)
    """
    beams = beams_factory.get_beams_low(frequency, soln_time)

    # NOTE: This ID mapping will not always work when the eb_ms file is
    # different. Should restrict the form of the eb_ms files allowed,
    # or preferably deprecate the eb_ms option.
    response = beams.array_response(direction=beams.beam_direction)

    # Tranpose to apply_ufunc expected dimensions order
    return response.transpose(1, 0, 2, 3)


def prediction_central_beams(
    gaintable: GainTable,
    beams_factory: BeamsFactory,
) -> GainTable:
    """
    Predict the central beam response for the provided gaintable configuration.

    This function calculates the primary beam response (central beam) for each
    antenna, frequency, and time step defined in the input `gaintable`. It
    uses the provided `beams_factory` to model the beam patterns.



    The calculation is parallelized over frequency chunks using `xarray` and
    `dask`. The resulting beam responses are stored as gains in a new
    `GainTable`.

    Parameters
    ----------
    gaintable : GainTable
        The template GainTable defining the time, frequency, and antenna
        structure for the prediction. The existing `gain` data in this table
        is ignored, but coordinates and attributes are preserved.
    beams_factory : BeamsFactory
        The factory object responsible for generating or retrieving the
        antenna beam models (e.g., based on antenna type, time, and frequency).

    Returns
    -------
    GainTable
        A new GainTable where the `gain` variable contains the predicted
        central beam responses (complex Jones matrices). The shape matches
        the input gaintable: (time, antenna, frequency, receptor1, receptor2).
    """
    # need to calculate central beam response across entire frequency
    frequency_xdr = xr.DataArray(
        gaintable.frequency, name="frequency_xdr"
    ).pipe(with_chunks, gaintable.chunksizes)

    response = xr.concat(
        [
            xr.apply_ufunc(
                _prediction_central_beams_ufunc,
                frequency_xdr,
                input_core_dims=[[]],
                output_core_dims=[("antenna", "receptor1", "receptor2")],
                dask="parallelized",
                output_dtypes=[
                    np.complex128,
                ],
                join="outer",
                dataset_join="outer",
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "antenna": gaintable.antenna.size,
                        "receptor1": gaintable.receptor1.size,
                        "receptor2": gaintable.receptor2.size,
                    }
                },
                kwargs={
                    "soln_time": val,
                    "beams_factory": beams_factory,
                },
            ).transpose("antenna", "frequency", "receptor1", "receptor2")
            for val in gaintable.time.data
        ],
        dim="time",
    )

    response = response.assign_coords(gaintable.gain.coords)
    response = response.assign_attrs(gaintable.gain.attrs)

    return gaintable.assign({"gain": response})
