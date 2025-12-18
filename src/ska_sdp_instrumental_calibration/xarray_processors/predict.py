import logging

import dask.array as da
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.visibility import Visibility

from ..data_managers.beams import BeamsFactory
from ..data_managers.sky_model import GlobalSkyModel, LocalSkyModel
from ._utils import with_chunks

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
) -> np.ndarray:
    """
    A helper function which bridges the gap between
    LocalSkyModel.create_vis and predict_vis_new functions

    :param uvw: (time, frequency, baselineid, spatial)
    :param frequency: (frequency,)
    :param station_rm: (nant,) or None
    :param polarisation: (polarisation,)
    :param antenna1: (nant,)
    :param antenna2: (nant,)
    :param phasecentre
    :param local_sky_model
    :param beams_factory
    :param output_dtype

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
    Predict visibilities from a Global Sky Model (Distributed).

    This function computes expected visibilities by evaluating the radio
    interferometry measurement equation. It uses a Global Sky Model (GSM) to
    generate local sky models for specific time steps and applies primary
    beam effects if a beam factory is provided.

    The prediction is distributed and chunked:

    1.  **Time:** Iterates over solution intervals defined by `soln_time` and
        `soln_interval_slices`.
    2.  **Frequency:** Handled via Dask chunking on the input visibility
        object.

    Parameters
    ----------
    vis
        The template visibility dataset. Its structure (time, frequency,
        baselines, UVW coordinates) determines the grid for the prediction.
    gsm
        The sky model source. The method `get_local_sky_model` is called for
        each solution time to retrieve sources above the horizon.
    soln_time
        Array of timestamps (float, seconds) representing the center of each
        solution interval.
    soln_interval_slices
        A list of slice objects. Each slice corresponds to a timestamp in
        `soln_time` and selects the range of time indices in `vis` that fall
        within that interval.
    beams_factory
        Factory to generate antenna primary beams. If None, the sky model is
        assumed to be multiplied by unity (no beam attenuation).
    station_rm
        Station-based Rotation Measures (RM) for ionospheric Faraday rotation
        simulation. If provided, it is passed to the prediction kernel.

    Returns
    -------
        A new Visibility object containing the predicted data in the `vis`
        variable. Metadata and coordinates are preserved from the input `vis`.

    Raises
    ------
    AssertionError
        If the number of solution times does not match the number of interval
        slices.
    """
    assert len(soln_interval_slices) == len(
        soln_time
    ), "lengths of soln_interval_slices and soln_time do not match"

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

        if isinstance(station_rm, da.Array):
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
        local_sky_model = gsm.get_local_sky_model(
            soln_time[idx], vis.configuration.location
        )

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
