import logging
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord

from ..data_managers.beams import BeamsFactory
from ..data_managers.processing_set import (
    get_array_location,
    get_baseline_antenna_indices,
    get_correlated_data,
    get_phase_centre,
    get_uvw_for_prediction,
    with_correlated_data,
)
from ..data_managers.schema.polarization import LINEAR_POLS
from ..data_managers.sky_model import GlobalSkyModel, LocalSkyModel
from ._utils import as_dask_scalar, unwrap_dask_scalar, with_chunks

logger = logging.getLogger(__name__)


def _predict_vis_ufunc(
    uvw: np.ndarray,
    local_sky_model: np.ndarray | LocalSkyModel,
    frequency: np.ndarray,
    beams_factory: np.ndarray | BeamsFactory | None,
    station_rm: np.ndarray | None,
    polarisation: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    phasecentre: SkyCoord,
    output_dtype: type = np.complex64,
) -> np.ndarray:
    """
    A helper function which bridges the gap between
    LocalSkyModel.create_vis and predict_vis_new functions

    :param uvw: (time, frequency, baseline_id, uvw_label)
    :param local_sky_model: LocalSkyModel, or 0-d object array wrapping it
    :param frequency: (frequency,)
    :param beams_factory: BeamsFactory or None, or 0-d object array
        wrapping it
    :param station_rm: (nant,) or None
    :param polarisation: (polarisation,)
    :param antenna1: (nant,)
    :param antenna2: (nant,)
    :param phasecentre
    :param output_dtype

    returns: (time, frequency, baseline_id, polarization)
    """
    # Need to remove extra frequency dimension from uvw
    uvw_shape = uvw.shape
    uvw = uvw.reshape(uvw_shape[0], *uvw_shape[2:])

    local_sky_model = unwrap_dask_scalar(local_sky_model)
    beams_factory = unwrap_dask_scalar(beams_factory)

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
        0, 2, 1, 3  # time, frequency, baseline_id, polarization
    )


def predict_vis(
    ms_xdt: xr.DataTree,
    gsm: GlobalSkyModel,
    soln_time: np.ndarray,
    soln_interval_slices: list[slice],
    beams_factory: Optional[BeamsFactory] = None,
    station_rm: Optional[xr.DataArray | da.Array] = None,
) -> xr.DataTree:
    """
    Predict visibilities of an MSv4 node from a Global Sky Model.

    This function computes expected visibilities by evaluating the radio
    interferometry measurement equation. It uses a Global Sky Model (GSM) to
    generate local sky models for specific time steps and applies primary
    beam effects if a beam factory is provided.

    The prediction is distributed and chunked:

    1.  **Time:** Iterates over solution intervals defined by `soln_time` and
        `soln_interval_slices`.
    2.  **Frequency:** Handled via Dask chunking of the correlated data.

    The MSv4 node supplies the prediction inputs:

    - UVW, negated to the sign convention of the prediction (see
      :func:`~.processing_set.get_uvw_for_prediction`),
    - baseline antenna indices, in ``antenna_xds`` order, with baselines in
      the antenna order of the data,
    - phase centre and array location.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single field and data group. Its structure
        (time, baselines, frequency, UVW) determines the grid for the
        prediction.
    gsm
        The sky model source. The method `get_local_sky_model` is called for
        each solution time to retrieve sources above the horizon.
    soln_time
        Centre of each solution interval, in seconds since the MJD epoch,
        e.g. the ``time`` of a gaintable created from ``ms_xdt``.
    soln_interval_slices
        A list of slice objects. Each slice corresponds to a timestamp in
        `soln_time` and selects the range of time indices in `ms_xdt` that
        fall within that interval.
    beams_factory
        Factory to generate antenna primary beams. If None, the sky model is
        assumed to be multiplied by unity (no beam attenuation).
    station_rm
        Station-based Rotation Measures (RM) for ionospheric Faraday rotation
        simulation, of shape ``(antenna_name,)`` in ``antenna_xds`` order.
        If provided, it is passed to the prediction kernel.

    Returns
    -------
        Copy of ``ms_xdt`` with the predicted visibilities as correlated
        data. Flags, weights and child datasets are those of ``ms_xdt``.

    Raises
    ------
    AssertionError
        If the number of solution times does not match the number of interval
        slices.
    ValueError
        If the polarizations are not XX XY YX YY, the only ones predicted.
    """
    assert len(soln_interval_slices) == len(
        soln_time
    ), "lengths of soln_interval_slices and soln_time do not match"

    correlated_data = get_correlated_data(ms_xdt)
    polarization = np.asarray(correlated_data.polarization.values, dtype=str)
    # The sky model is predicted as linear XX XY YX YY, and only the number
    # of polarizations is passed to it. So check the labels and their order.
    if tuple(polarization) != LINEAR_POLS:
        raise ValueError(
            f"Prediction requires polarizations {list(LINEAR_POLS)}, "
            f"got {polarization.tolist()}"
        )

    antenna1, antenna2 = get_baseline_antenna_indices(ms_xdt)
    uvw = get_uvw_for_prediction(ms_xdt)
    array_location = get_array_location(ms_xdt)

    common_input_args = []
    common_input_core_dims = []

    input_kwargs = dict(
        polarisation=polarization,
        antenna1=antenna1,
        antenna2=antenna2,
        phasecentre=get_phase_centre(ms_xdt),
        output_dtype=correlated_data.dtype,
    )

    # Process frequency
    # Convert frequency to a chunked dask array, with the frequency chunks of
    # the correlated data. Chunks are read from the data only, as
    # DataArray.chunksizes also checks time coordinates, which may be
    # chunked differently (e.g. on predicted model visibilities).
    data_chunks = dict(zip(correlated_data.dims, correlated_data.chunks or ()))
    frequency_xdr = xr.DataArray(
        correlated_data.frequency, name="frequency_xdr"
    ).pipe(with_chunks, data_chunks)
    common_input_args.append(frequency_xdr)
    common_input_core_dims.append([])

    # Process beams_factory
    # Wrap it as a single graph key, so that it is tokenized and
    # transferred once, instead of being embedded in every task.
    # Always passed positionally (even if None) to keep argument order.
    common_input_args.append(
        as_dask_scalar(beams_factory, name="beams_factory")
    )
    common_input_core_dims.append([])

    # Process station_rm
    if station_rm is not None:
        # Ensure that it is not chunked across any dim
        # It can still be a dask array

        if isinstance(station_rm, da.Array):
            # In antenna_xds order, like gaintables
            antenna_names = ms_xdt["antenna_xds"].ds.antenna_name.values
            station_rm = xr.DataArray(
                station_rm, coords={"antenna_name": antenna_names}
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
            soln_time[idx], array_location
        )

        predicted_per_soln_time: xr.DataArray = xr.apply_ufunc(
            _predict_vis_ufunc,
            uvw.isel(time=slc),
            as_dask_scalar(local_sky_model, name="local_sky_model"),
            *common_input_args,
            input_core_dims=[
                ["baseline_id", "uvw_label"],
                [],
                *common_input_core_dims,
            ],
            output_core_dims=[
                ["baseline_id", "polarization"],
            ],
            dask="parallelized",
            output_dtypes=[correlated_data.dtype],
            dask_gufunc_kwargs=dict(
                output_sizes={
                    "baseline_id": correlated_data.baseline_id.size,
                    "polarization": polarization.size,
                }
            ),
            kwargs=input_kwargs,
        )
        predicted_across_soln_time.append(predicted_per_soln_time)

    predicted: xr.DataArray = xr.concat(predicted_across_soln_time, dim="time")

    return with_correlated_data(ms_xdt, predicted)
