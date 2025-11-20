import logging

import dask.array as da
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.visibility import Visibility

from ..data_managers.local_sky_model import GlobalSkyModel, LocalSkyModel
from ..processing_tasks.predict_model.beams import BeamsFactory
from ..workflow.utils import with_chunks

logger = logging.getLogger(__name__)


def _predict_vis_ufunc(
    uvw: np.ndarray,
    frequency: np.ndarray,
    station_rm: np.ndarray | None,
    polarisation: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    phasecentre: SkyCoord,
    local_sky_model: LocalSkyModel,
    beams_factory: BeamsFactory = None,
    output_dtype: type = np.complex64,
):
    """
    A helper function which bridges the gap between
    LocalSkyModel.create_vis and predict_vis_new functions

    :param uvw: (time, frequency, baselineid, spatial)
    :param frequency: (frequency,)
    :param station_rm: (nant,) or None
    :param polarisation: (polarisation,)
    :param antenna1: (nant,)
    :param antenna2: (nant,)
    :param configuration: object
    :param phasecentre: object
    :param lsm: Component List containing the local sky model
    :param beam_type: str
        Type of beam model to use. Default is "everybeam". If set
        to None, no beam will be applied.
    :param eb_ms: str
        Measurement set need to initialise the everybeam telescope.
        Required if beam_type is "everybeam".
    :param soln_time: float
        "Solution time" value of the gain solution. Used for initialising Beams
        for that current time slice. Required if beam_type is "everybeam".
        Must be a single time value.
    :param output_dtype: Type

    returns: (time, frequency, baselineid, polarisation)
    """
    # Need to remove extra frequency dimension from uvw
    uvw_shape = uvw.shape
    uvw = uvw.reshape(uvw_shape[0], *uvw_shape[2:])

    return local_sky_model.create_vis(
        uvw,
        frequency,
        polarisation,
        phasecentre,
        antenna1,
        antenna2,
        beams_factory,
        station_rm,
        output_dtype,
    ).transpose(
        0, 2, 1, 3  # time, frequency, baselineid, polarisation
    )


def predict_vis(
    vis: Visibility,
    gsm: GlobalSkyModel,
    soln_time: np.ndarray,
    soln_interval_slices: list[slice],
    beams_factory: BeamsFactory = None,
    station_rm: xr.DataArray = None,
) -> Visibility:
    """
    Distributed Visibility predict.
    Supports chunking across frequency and time.
    """
    assert len(soln_interval_slices) == len(soln_time), (
        "lengths of " "soln_interval_slices and soln_time do not match"
    )

    common_input_args = []
    common_input_core_dims = []

    input_kwargs = dict(
        polarisation=vis.polarisation,
        antenna1=vis.antenna1,
        antenna2=vis.antenna2,
        phasecentre=vis.phasecentre,
        beams_factory=beams_factory,
        output_dtype=vis.vis.dtype,
    )

    # Process frequency
    # Convert frequency to a chunked dask array
    frequency_xdr = xr.DataArray(vis.frequency, name="frequency_xdr").pipe(
        with_chunks, vis.chunksizes
    )
    common_input_args.append(frequency_xdr)
    common_input_core_dims.append([])

    # Process station_rm
    if station_rm is not None:
        # Ensure that it is not chunked across any dim
        # It can still be a dask array

        if type(station_rm) is da.Array:
            # "id" is a coordinate from Configuration dataset
            station_rm = xr.DataArray(
                station_rm, coords={"id": np.arange(len(station_rm))}
            )
        station_rm = station_rm.chunk(-1)
        common_input_args.append(station_rm)
        common_input_core_dims.append(list(station_rm.dims))

    else:
        input_kwargs["station_rm"] = None

    # Call predict ufunc, once per solution interval
    predicted_across_soln_time = []
    for idx, slc in enumerate(soln_interval_slices):
        local_sky_model = gsm.get_local_sky_model(soln_time[idx])

        predicted_per_soln_time: xr.DataArray = xr.apply_ufunc(
            _predict_vis_ufunc,
            vis.uvw.isel(time=slc),
            *common_input_args,
            input_core_dims=[
                ["baselineid", "spatial"],
                *common_input_core_dims,
            ],
            output_core_dims=[
                ["baselineid", "polarisation"],
            ],
            dask="parallelized",
            output_dtypes=[vis.vis.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes={
                    "baselineid": vis.baselineid.size,
                    "polarisation": vis.polarisation.size,
                }
            ),
            kwargs=dict(
                **input_kwargs,
                local_sky_model=local_sky_model,
            ),
        )
        predicted_per_soln_time = predicted_per_soln_time.transpose(
            "time", "baselineid", "frequency", "polarisation"
        )
        predicted_across_soln_time.append(predicted_per_soln_time)

    predicted: xr.DataArray = xr.concat(predicted_across_soln_time, dim="time")

    predicted = predicted.assign_attrs(vis.vis.attrs)

    return vis.assign({"vis": predicted})
