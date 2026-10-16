import logging

import numpy as np
import xarray as xr

from ..data_managers.beams import BeamsFactory
from ..data_managers.schema.calibration_set import GainCalibrationSetXds
from ..numpy_processors._utils import jones_to_pol
from ._utils import as_dask_scalar, unwrap_dask_scalar, with_chunks

logger = logging.getLogger(__name__)


def _prediction_central_beams_ufunc(
    frequency: np.ndarray,
    beams_factory: np.ndarray | BeamsFactory,
    soln_time: float,
):
    """
    frequency: np.ndarray, (frequency,)
    beams_factory: BeamsFactory, or 0-d object array wrapping it
    soln_time: float

    Returns
    -------
    np.ndarray (frequency, antenna, npol)
    """
    beams_factory = unwrap_dask_scalar(beams_factory)
    beams = beams_factory.get_beams_low(frequency, soln_time)

    # NOTE: This ID mapping will not always work when the eb_ms file is
    # different. Should restrict the form of the eb_ms files allowed,
    # or preferably deprecate the eb_ms option.
    response = beams.array_response(direction=beams.beam_direction)

    # Tranpose to apply_ufunc expected dimensions order
    return jones_to_pol(response.transpose(1, 0, 2, 3))


def prediction_central_beams(
    gaintable: GainCalibrationSetXds,
    beams_factory: BeamsFactory,
) -> GainCalibrationSetXds:
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
    gaintable : GainCalibrationSetXds
        The template GainTable defining the time, frequency, and antenna
        structure for the prediction. The existing `gain` data in this table
        is ignored, but coordinates and attributes are preserved.
    beams_factory : BeamsFactory
        The factory object responsible for generating or retrieving the
        antenna beam models (e.g., based on antenna type, time, and frequency).

    Returns
    -------
    GainCalibrationSetXds
        A new GainTable where the `gain` variable contains the predicted
        central beam responses (complex Jones matrices). The shape matches
        the input gaintable: (time, antenna, frequency, polarization).
    """
    # need to calculate central beam response across entire frequency
    frequency_xdr = xr.DataArray(
        gaintable.frequency, name="frequency_xdr"
    ).pipe(with_chunks, gaintable.chunksizes)

    # Single graph key, tokenized and transferred once
    beams_factory_xdr = as_dask_scalar(beams_factory, name="beams_factory")

    response = xr.concat(
        [
            xr.apply_ufunc(
                _prediction_central_beams_ufunc,
                frequency_xdr,
                beams_factory_xdr,
                input_core_dims=[[], []],
                output_core_dims=[("antenna_name", "polarization")],
                dask="parallelized",
                output_dtypes=[
                    np.complex128,
                ],
                join="outer",
                dataset_join="outer",
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "antenna_name": gaintable.antenna_name.size,
                        "polarization": gaintable.polarization.size,
                    }
                },
                kwargs={
                    "soln_time": val,
                },
            ).transpose("antenna_name", "frequency", "polarization")
            for val in gaintable.time.data
        ],
        dim="time",
    )

    response = response.assign_coords(gaintable.CALPARAM_GAIN.coords)
    response = response.assign_attrs(gaintable.CALPARAM_GAIN.attrs)

    return gaintable.assign({"CALPARAM_GAIN": response})
