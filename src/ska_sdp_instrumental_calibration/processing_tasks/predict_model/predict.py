import logging
from typing import Any, Mapping, Optional

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
    dataarray : xr.DataArray
        Input DataArray (can be Dask-backed or not).
    chunks : dict
        Dictionary mapping dimension names to chunk sizes.

    Returns
    -------
    xr.DataArray
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
    chunks: dict | None = None,
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
    frequency_xdr: xr.DataArray
        Frequency list in Hz ("frequency")
    polarisation_coord: xr.DataArray
        Polarisation ("pol")
    chunks: dict | None, default=None
        Chunks information required for chunking skycomponent flux DataArray.

    Returns
    --------
    SkyComponent
        A SkyComponent.
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


def generate_rotation_matrices(
    rm: da.Array,
    frequency_xdr: xr.DataArray,
    antenna_id_coord: xr.DataArray,
    chunks: Mapping[Any, tuple[int, ...]] = {},
    output_dtype: type = np.float64,
) -> xr.DataArray:
    """Generate station rotation matrices from RM values.

    Parameters
    ----------
    rm: da.Array
        1D dask array of rotation measure values [nstation]
    frequency_xdr: xr.DataArray
        1D array of frequency values [nfrequency].

    Returns
    -------
    xr.DataArray
        4D array of rotation matrices: [nstation, nfrequency, 2, 2].
    """
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
    """Calculated visibility amplitude tapers for Gaussian components.

    Note: this needs to be tested. Generate, image and fit a model component?

    Parameters
    ----------
    scaled_u: np.ndarray
        ("time", "frequency", "baselineid")
    scaled_v: np.ndarray
        ("time", "frequency", "baselineid")
    params: dict
        Dictionary of shape params {bmaj, bmin, bpa} in degrees.

    Returns
    --------
    np.ndarray
        visibility tapers ([time, frequency, baselineid]).
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
    Quick 'n dirty numpy-based predict for local testing without sdp.func.

    Parameters
    ----------
    scaled_u: np.ndarray
         ("time", "frequency", "baselineid")
    scaled_v: np.ndarray
         ("time", "frequency", "baselineid")
    scaled_w: np.ndarray
         ("time", "frequency", "baselineid")
    skycomponent_flux: np.ndarray
        (frequency, polarisation)
    skycomponent: SkyComponent
    phase_centre: SkyCoord
        SkyCoord (ICRS): (ra, dec) in deg
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
    comp_vis: np.ndarray,
    correction: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
):
    """
    Apply correction on component visibilities.

    Parameters
    ----------
    comp_vis: np.ndarray
        ("time", "freq", "baselineid", "polarisation")

    correction: np.ndarray
        ("freq", "id", "x", "y")

    antenna1: np.ndarray
        ("baselineid")

    antenna2: np.ndarray
        ("baselineid")


    Returns
    -------
    np.ndarray
        ("time", "freq", "baselineid", "polarisation")
        Corrected component visibilities.
    """
    return np.einsum(  # pylint: disable=too-many-function-args
        "fbpx,tfbxy,fbqy->tfbpq",
        correction[:, antenna1, :, :],
        comp_vis.reshape(comp_vis.shape[:3] + (2, 2)),
        correction[:, antenna2, :, :].conj(),
    ).reshape(comp_vis.shape)


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
    """Predict model visibilities from a SkyComponent List.

    Parameters
    ----------
    visibility: xr.DataArray
        ('time', 'baselineid', 'frequency', 'polarisation')

    uvw: xr.DataArray
        ('time', 'baselineid', 'spatial')

    datetime: xr.DataArray
        ('time',)

    configuration: Configuration
        ('id', 'spatial')

    antenna1: xr.DataArray
        ('baselineid',)

    antenna2: xr.DataArray
        ('baselineid',)

    components: list[Component]
        LSM components.

    phase_centre: SkyCoord
        SkyCoord (ICRS): (ra, dec) in deg

    station_rm: Optional[da.Array], default=None
        Station rotation measure values.

    beam_type: Optional[str] = "everybeam"
        Type of beam model to use. Default is "everybeam". If set
        to None, no beam will be applied.

    eb_ms: Optional[str], default=None
        Measurement set need to initialise the everybeam telescope.
        Required if beam_type is "everybeam".

    eb_coeffs: Optional[str], default=None
        Everybeam coeffs datadir containing beam coefficients.
        Required if beam_type is "everybeam".

    Returns
    -------
    Visibilities: xr.DataArray
        Model Visibilities predicted from components.
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
        if time and beams:
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
