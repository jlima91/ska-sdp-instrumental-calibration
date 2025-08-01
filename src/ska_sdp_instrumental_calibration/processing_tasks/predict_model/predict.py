import logging
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
from astropy import constants as const
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.processing_tasks.lsm import (
    Component,
    deconvolve_gaussian,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict_model.beams import (  # noqa: E501
    create_beams,
)

logger = logging.getLogger()


def with_chunks(dataarray: xr.DataArray, chunks: dict) -> xr.DataArray:
    """
    Rechunk a DataArray along dimensions specified in `chunks` dict.

    Parameters
    ----------
    dataarray : xarray.DataArray
        Input DataArray (can be Dask-backed or not).
    chunks: dict
        A dictionary mapping dimension names to chunk sizes.

    Returns
    -------
    xarray.DataArray
        Rechunked DataArray if applicable.
    """
    relevant_chunks = {
        dim: chunks[dim] for dim in dataarray.dims if dim in chunks
    }

    return dataarray.chunk(relevant_chunks) if relevant_chunks else dataarray


def convert_comp_to_skycomponent(
    comp: Component,
    frequency_xdr: xr.DataArray,
    polarisation_coord: xr.DataArray,
    chunks: dict = None,
) -> SkyComponent:
    """
    Convert the LocalSkyModel to a list of SkyComponent.

    All sources are unpolarised and specified in the linear polarisation frame
    using XX = YY = Stokes I/2.

    Function :func:`~deconvolve_gaussian` is used to deconvolve the MWA
    synthesised beam from catalogue shape parameters of each component.
    Components with non-zero widths after this process are stored with
    shape = "GAUSSIAN". Otherwise shape = "POINT".

    Parameters
    ----------
    comp: Component
        Component to convert to SkyComponent

    frequency_xdr: xarray.DataArray
        The frequency dataarray whose data is a 1D dask/numpy array
        containing all frequency values.

        Dimensions: ``[frequency,]``

    polarisation_coord: xarray.DataArray
        The polarisation coordinate of the input Visbility dataset.
        Its data must be a 1D numpy array, containing polarisation values.
        For example: ``["XX", "XY", "YX", "YY"]``

        Dimensions: ``[polarisation,]``

    chunks: dict, optional
        A dictionary mapping dimension names to chunk sizes.

    Returns
    -------
    SkyComponent
        A SkyComponent. Its ``flux`` attribute is a xarray datarray with
        dimensions ``[frequency, polarisation]``.
    """
    chunks = chunks or {}

    if not np.array_equal(["XX", "XY", "YX", "YY"], polarisation_coord.data):
        raise ValueError(
            "Only polarisation ['XX', 'XY', 'YX', 'YY'] is supported."
        )
    pol_frame = PolarisationFrame("linear")

    xx_yy_flux_xdr = (comp.flux / 2) * np.power(
        (frequency_xdr / comp.ref_freq), comp.alpha
    )

    flux_xdr = xx_yy_flux_xdr * xr.DataArray(
        da.asarray([1, 0, 0, 1], dtype=frequency_xdr.dtype),
        coords={"polarisation": polarisation_coord},
    ).pipe(with_chunks, chunks)

    # Deconvolve synthesised beam from fitted shape parameters.
    smaj, smin, spa = deconvolve_gaussian(comp)
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

    skycomponent = SkyComponent(
        direction=SkyCoord(
            ra=comp.RAdeg,
            dec=comp.DEdeg,
            unit="deg",
        ),
        frequency=frequency_xdr.frequency.data,
        name=comp.name,
        flux=np.zeros_like(flux_xdr.data),
        polarisation_frame=pol_frame,
        shape=shape,
        params=params,
    )

    # TODO: Remove this hack once Skycomponent natively supports dask arrays
    skycomponent.flux = flux_xdr

    return skycomponent


# Revisit for MSv4 transition
def generate_rotation_matrices(
    rm: da.Array,
    frequency_xdr: xr.DataArray,
    antenna_id_coord: xr.DataArray,
    chunks: dict = None,
    output_dtype: type = np.float64,
) -> xr.DataArray:
    """Generate station rotation matrices from RM values.

    Parameters
    ----------
    rm: dask.Array
        1D dask array of rotation measure values.
        The size of the array is equal to number of stations.

        Dimensions: ``[id,]``

    frequency_xdr: xarray.DataArray
        The frequency dataarray whose data is a 1D dask/numpy array
        containing all frequency values.

        Dimensions: ``[frequency,]``

    antenna_id_coord: xarray.DataArray
        The ``id`` coordinate of the ``configuration`` dataset. This dataset
        contains the antenna configuration information.
        Its data must be a 1D numpy array, containing antenna ids.

        Dimensions: ``[id,]``

    chunks: dict, optional
        A dictionary mapping dimension names to chunk sizes.

    output_dtype: type, default: np.float64
        Datatype of the output rotation matrix dataarray

    Returns
    -------
    xarray.DataArray
        4D array of rotation matrices. The size of
        'x' and 'y' dimensions will always be 2.

        Dimensions: ``[frequency, id, x, y]``
    """
    chunks = chunks or {}

    rm_xdr = xr.DataArray(rm, coords={"id": antenna_id_coord}).pipe(
        with_chunks, chunks
    )

    lambda_sq_xdr = np.power(
        (const.c.value / frequency_xdr), 2  # pylint: disable=no-member
    )

    phi_xdr = lambda_sq_xdr * rm_xdr

    cos_val = np.cos(phi_xdr)
    sin_val = np.sin(phi_xdr)

    identity = np.array([[1, 0], [0, 1]])
    inverse = np.array([[0, -1], [1, 0]])

    rot_array_da = (
        cos_val.data[:, :, np.newaxis, np.newaxis] * identity
        + sin_val.data[:, :, np.newaxis, np.newaxis] * inverse
    )

    return (
        xr.DataArray(
            rot_array_da,
            dims=["frequency", "id", "x", "y"],
            coords={
                "frequency": frequency_xdr.frequency,
                "id": antenna_id_coord,
            },
        )
        .pipe(with_chunks, chunks)
        .astype(output_dtype, copy=False)
    )


def gaussian_tapers_ufunc(
    scaled_u: np.ndarray,
    scaled_v: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """
    Calculate visibility amplitude tapers for Gaussian components.

    Note: this needs to be tested. Generate, image and fit a model component?

    Parameters
    ----------
    scaled_u: np.ndarray
        The 3D array with "u" values scaled across frequencies.

        Dimensions: ``[time, frequency, baselineid]``

    scaled_v: np.ndarray
        The 3D array with "v" values scaled across frequencies.

        Dimensions: ``[time, frequency, baselineid]``

    params: dict
        Dictionary of shape params ``{bmaj, bmin, bpa}``
        in degrees.

    Returns
    -------
    np.ndarray
        Visibility tapers

        Dimensions: ``[time, frequency, baselineid]``
    """
    scale = -(np.pi * np.pi) / (4 * np.log(2.0))
    # Rotate baselines to the major/minor axes:
    bpa = params["bpa"] * np.pi / 180
    bmaj = params["bmaj"] * np.pi / 180
    bmin = params["bmin"] * np.pi / 180

    up = np.cos(bpa) * scaled_u + np.sin(bpa) * scaled_v
    vp = -np.sin(bpa) * scaled_u + np.cos(bpa) * scaled_v

    return np.exp((bmaj * bmaj * up * up + bmin * bmin * vp * vp) * scale)


def dft_skycomponent_ufunc(
    scaled_u: np.ndarray,
    scaled_v: np.ndarray,
    scaled_w: np.ndarray,
    skycomponent_flux: np.ndarray,
    skycomponent: SkyComponent,
    phase_centre: SkyCoord,
) -> np.ndarray:
    """
    Predict visibilities for a single skycomponent.
    This is a numpy ufunc, compatible with
    :py:func:`xarray.apply_ufunc` operation.
    The data can be chunked across
    ``time`` and ``frequency`` dimensions.

    Parameters
    ----------
    scaled_u: numpy.ndarray
        The 3D array with "u" values scaled across frequencies.

        Dimensions: ``[time, frequency, baselineid]``

    scaled_v: numpy.ndarray
        The 3D array with "v" values scaled across frequencies.

        Dimensions: ``[time, frequency, baselineid]``

    scaled_w: numpy.ndarray
        The 3D array with "w" values scaled across frequencies.

        Dimensions: ``[time, frequency, baselineid]``

    skycomponent_flux: numpy.ndarray
        The flux value of skycomponent.
        This value is passed seperately than the actual
        "skycomponent" since this array can be distributed
        per chunk by an apply_ufunc like operation.

        Dimensions: ``[frequency, polarisation]``

    skycomponent: SkyComponent
        The skycomponent to predict visibility for

    phase_centre: SkyCoord
        The beam phase center value. This is an astropy
        skycoord object.

    Returns
    -------
    np.ndarray
        A 4D numpy array containing predicted visibilities
        for the given skycomponent.

        Dimensions: ``[time, frequency, baselineid, polarisation]``
    """

    # Get coordaintes of phase centre
    ra0 = phase_centre.ra.radian
    phase_centre.dec
    cdec0 = np.cos(phase_centre.dec.radian)
    sdec0 = np.sin(phase_centre.dec.radian)

    cdec = np.cos(skycomponent.direction.dec.radian)
    sdec = np.sin(skycomponent.direction.dec.radian)
    cdra = np.cos(skycomponent.direction.ra.radian - ra0)
    l_comp = cdec * np.sin(skycomponent.direction.ra.radian - ra0)
    m_comp = sdec * cdec0 - cdec * sdec0 * cdra
    n_comp = sdec * sdec0 + cdec * cdec0 * cdra

    comp_data = np.exp(
        -2j
        * np.pi
        * (scaled_u * l_comp + scaled_v * m_comp + scaled_w * (n_comp - 1))
    )

    if skycomponent.shape == "GAUSSIAN":
        comp_data = comp_data * gaussian_tapers_ufunc(
            scaled_u, scaled_v, skycomponent.params
        )

    return np.einsum(
        "tfb,fp->tfbp",
        comp_data,
        skycomponent_flux,
    )


def correct_comp_vis_ufunc(
    visibility: np.ndarray,
    correction: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
):
    """
    Apply correction on visibilities.
    This is a numpy ufunc, compatible to be used with
    :py:func:`xarray.apply_ufunc`.

    Parameters
    ----------
    visibility: np.ndarray
        Visibility data

        Dimensions: ``[time, frequency, baselineid, polarisation]``

    correction: np.ndarray
        Corrections to apply on visibility data.

        Dimensions: ``[frequency, id, x, y]``

    antenna1: np.ndarray
        The indices of the first antennas in a baseline pair

        Dimensions: ``[baselineid,]``

    antenna2: np.ndarray
        The indices of the second antennas in a baseline pair

        Dimensions: ``[baselineid,]``

    Returns
    -------
    np.ndarray
        Corrected visibilities.

        Dimensions: ``[time, frequency, baselineid, polarisation]``
    """
    return np.einsum(  # pylint: disable=too-many-function-args
        "fbpx,tfbxy,fbqy->tfbpq",
        correction[:, antenna1, :, :],
        visibility.reshape(visibility.shape[:3] + (2, 2)),
        correction[:, antenna2, :, :].conj(),
    ).reshape(visibility.shape)


def predict_vis(
    visibility: xr.DataArray,
    uvw: xr.DataArray,
    datetime: xr.DataArray,
    configuration: Configuration,
    antenna1: xr.DataArray,
    antenna2: xr.DataArray,
    components: list[Component],
    phase_centre: SkyCoord,
    station_rm: Optional[da.Array] = None,
    beam_type: Optional[str] = "everybeam",
    eb_ms: Optional[str] = None,
    eb_coeffs: Optional[str] = None,
) -> xr.DataArray:
    """
    Predict model visibilities from local sky model represented
    as a list of components.

    Parameters
    ----------
    visibility: xarray.DataArray
        The visibitlity dataarray.
        This is only used to ensure that predicted visibitlies
        have same shape and datatype as this one.
        The actual data is never accessed.
        The dimensions can be in any order, but the preferred
        order is given below, which ensures less reshape/tranpose
        operations during parallel computing.

        Dimensions: ``[time, frequency, baselineid, polarisation]``

    uvw: xarray.DataArray
        The uvw dataarray from input dataset.

        Dimensions: ``[time, baselineid, spatial]``

    datetime: xarray.DataArray
        The datetime dataarray from Visibility dataset

        Dimensions: ``[time]``

    configuration: Configuration
        The dataset containing antenna configuration information.

    antenna1: xarray.Dataarray
        The indices of the first antennas in a baseline pair

        Dimensions: ``[baselineid,]``

    antenna2: xarray.Dataarray
        The indices of the second antennas in a baseline pair

        Dimensions: ``[baselineid,]``

    components: list of Component
        List of components in local sky model.

    phase_centre: SkyCoord
        The beam phase center value. This is an astropy
        skycoord object.

    station_rm: dask.Array, optional
        Station rotation measure values.

    beam_type: str, default: "everybeam"
        Type of beam model to use. Default is "everybeam". If set
        to None, no beam will be applied.

    eb_ms: str, optional
        Measurement set need to initialise the everybeam telescope.
        Required if beam_type is "everybeam".

    eb_coeffs: str, optional
        Everybeam coeffs datadir containing beam coefficients.
        Required if beam_type is "everybeam".

    Returns
    -------
    xarray.DataArray
        Model visibilities predicted from components. This has same dimension
        as the input ``visibility`` dataarray.
    """
    chunks = visibility.chunksizes

    frequency_xdr = xr.DataArray(visibility.frequency).pipe(
        with_chunks, chunks
    )

    prediced_vis = xr.zeros_like(visibility)

    scaled_uvw = (
        uvw * frequency_xdr / const.c.value  # pylint: disable=no-member
    )

    scaled_u = scaled_uvw.sel(spatial="u")
    scaled_v = scaled_uvw.sel(spatial="v")
    scaled_w = scaled_uvw.sel(spatial="w")

    time = None
    beams = None
    # Set up the beam model
    if beam_type == "everybeam":
        logger.info("Using EveryBeam model in predict")

        if not all(["XX", "XY", "YX", "YY"] == visibility.polarisation.data):
            raise ValueError(
                "Beams are supported only for linear polarisation."
            )

        if eb_coeffs is None or eb_ms is None:
            raise ValueError("eb_coeffs and eb_ms required for everybeam")

        # TODO: Delay this somehow
        logger.warning(
            "Triggering eager compute for mean of datetime values."
            " This is required for beam calculation."
        )
        # TODO: Verify: Just use the beam near the middle of the scan?
        time = np.mean(Time(datetime.data))
        beams = create_beams(
            time, frequency_xdr, configuration, phase_centre, eb_coeffs, eb_ms
        )

    else:
        logger.info("No beam model used in predict")

    rot_array_xdr = None
    # Set up the Faraday rotation model
    if station_rm is not None:
        if len(station_rm) != len(configuration.id):
            raise ValueError("unexpected length for station_rm")

        rot_array_xdr = generate_rotation_matrices(
            station_rm,
            frequency_xdr,
            configuration.id,
            chunks=chunks,
            output_dtype=visibility.dtype,
        ).chunk({"x": -1, "y": -1})

    for comp in components:

        skycomponent = convert_comp_to_skycomponent(
            comp, frequency_xdr, visibility.polarisation, chunks
        )

        response = None
        # Apply beam distortions and add to combined model visibilities
        if not (time is None or beams is None):
            # Check component direction
            altaz = skycomponent.direction.transform_to(
                AltAz(obstime=time, location=beams.array_location)
            )
            if altaz.alt.degree < 0:
                logger.warning(
                    "LSM component [%s] below horizon", skycomponent.name
                )
                continue

            beam_array_resp_xdr = (
                beams.array_response(
                    direction=skycomponent.direction,
                    frequency_xdr=frequency_xdr,
                    time=time,
                )
                .assign_coords({"id": configuration.id})
                .pipe(with_chunks, chunks)
            )

            if rot_array_xdr is not None:
                response = xr.apply_ufunc(
                    np.matmul,
                    beam_array_resp_xdr,
                    rot_array_xdr,
                    input_core_dims=[
                        ["x", "y"],
                        ["x", "y"],
                    ],
                    output_core_dims=[("x", "y")],
                    dask="parallelized",
                    output_dtypes=[beam_array_resp_xdr.dtype],
                )
            else:
                response = beam_array_resp_xdr
        else:
            response = rot_array_xdr

        comp_vis = xr.apply_ufunc(
            dft_skycomponent_ufunc,
            scaled_u,
            scaled_v,
            scaled_w,
            skycomponent.flux,
            input_core_dims=[
                [
                    "baselineid",
                ],
                [
                    "baselineid",
                ],
                [
                    "baselineid",
                ],
                ["polarisation"],
            ],
            output_core_dims=[("baselineid", "polarisation")],
            dask="parallelized",
            output_dtypes=[visibility.dtype],
            kwargs={
                "skycomponent": skycomponent,
                "phase_centre": phase_centre,
            },
        )

        if response is not None:
            prediced_vis = prediced_vis + xr.apply_ufunc(
                correct_comp_vis_ufunc,
                comp_vis,
                response,
                antenna1,
                antenna2,
                input_core_dims=[
                    ["baselineid", "polarisation"],
                    ["id", "x", "y"],
                    ["baselineid"],
                    ["baselineid"],
                ],
                output_core_dims=[("baselineid", "polarisation")],
                dask="parallelized",
                output_dtypes=[visibility.dtype],
            )
        else:
            prediced_vis = prediced_vis + comp_vis

    return prediced_vis
