import logging

import numpy as np
import xarray as xr
from astropy import constants as const
from ska_sdp_datamodels.calibration import GainTable
from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.visibility import Visibility
from ska_sdp_func_python.calibration.ionosphere_solvers import (
    get_param_count,
    set_cluster_maps,
    set_coeffs_and_params,
)

__all__ = ["run_ionospheric_solver", "IonosphericSolver"]

logger = logging.getLogger(__name__)

import logging

import h5py
import numpy as np
import xarray as xr
from astropy.io import fits
from astropy.time import Time

LOGGER = logging.getLogger(__name__)

def _project_stations_to_uvw(
    station_xyz_m: np.ndarray,
    times_mjd: np.ndarray,
    phasecentre,
    longitude_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project geocentric station XYZ positions onto the UV plane.

    Parameters
    ----------
    station_xyz_m
        Geocentric station coordinates with shape ``(antenna, 3)`` in metres.
    times_mjd
        Observation times as Modified Julian Dates.
    phasecentre
        Observation phase centre as an Astropy SkyCoord.
    longitude_rad
        Array longitude in radians.

    Returns
    -------
    station_u_m, station_v_m
        Arrays with shape ``(time, antenna)`` in metres.

    Notes
    -----
    Station positions are centred on the array before projection because
    only relative positions are required for station-based phase gains.
    """
    station_xyz_m = np.asarray(station_xyz_m, dtype=float)
    times_mjd = np.asarray(times_mjd, dtype=float)

    if station_xyz_m.ndim != 2 or station_xyz_m.shape[1] != 3:
        raise ValueError(
            "station_xyz_m must have shape (antenna, 3), "
            f"received {station_xyz_m.shape}."
        )

    # Place the array centre at the TEC-screen origin.
    xyz = station_xyz_m - station_xyz_m.mean(axis=0)

    x_coord = xyz[:, 0]
    y_coord = xyz[:, 1]
    z_coord = xyz[:, 2]

    observation_times = Time(
        times_mjd,
        format="mjd",
        scale="utc",
    )

    local_sidereal_time = observation_times.sidereal_time(
        "apparent",
        longitude=longitude_rad,
    ).rad

    hour_angle = local_sidereal_time - phasecentre.icrs.ra.rad
    declination = phasecentre.icrs.dec.rad

    sin_ha = np.sin(hour_angle)[:, np.newaxis]
    cos_ha = np.cos(hour_angle)[:, np.newaxis]

    sin_dec = np.sin(declination)
    cos_dec = np.cos(declination)

    station_u_m = (
        sin_ha * x_coord[np.newaxis, :]
        + cos_ha * y_coord[np.newaxis, :]
    )

    station_v_m = (
        -sin_dec * cos_ha * x_coord[np.newaxis, :]
        + sin_dec * sin_ha * y_coord[np.newaxis, :]
        + cos_dec * z_coord[np.newaxis, :]
    )

    return station_u_m, station_v_m


def _screen_indices_for_times(
    solution_times: np.ndarray,
    observation_start_time: float,
    screen_time_interval_s: float,
    number_of_screens: int,
) -> np.ndarray:
    """
    Map solution timestamps to nearest TEC-screen slices.

    This reproduces OSKAR's elapsed-time nearest-slice behaviour.
    """
    if number_of_screens <= 0:
        raise ValueError("The TEC screen contains no time planes.")

    if screen_time_interval_s <= 0:
        # OSKAR advances one screen per simulation time step when the interval
        # is unavailable. Here, one screen is selected per gain solution.
        return np.clip(
            np.arange(solution_times.size),
            0,
            number_of_screens - 1,
        )

    elapsed_time_s = (
        np.asarray(solution_times, dtype=float)
        - float(observation_start_time)
    )

    indices = np.floor(
        elapsed_time_s / screen_time_interval_s + 0.5
    ).astype(int)

    return np.clip(
        indices,
        0,
        number_of_screens - 1,
    )


def _sample_phasecentre_tec(
    screen: np.ndarray,
    station_u_m: np.ndarray,
    station_v_m: np.ndarray,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> np.ndarray:
    """
    Sample one TEC screen for all stations at the phase centre.

    At the phase centre, ``l = m = 0``, so the pierce-point coordinates
    depend only on the projected station U and V positions.

    Parameters
    ----------
    screen
        Two-dimensional TEC screen with shape ``(y, x)``.
    station_u_m, station_v_m
        Projected station coordinates with shape ``(antenna,)``.
    pixel_size_x_m, pixel_size_y_m
        Screen pixel sizes in metres.

    Returns
    -------
    np.ndarray
        TEC values with shape ``(antenna,)``.
    """
    screen = np.asarray(screen)

    if screen.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional TEC screen, got {screen.shape}."
        )

    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError(
            "TEC-screen pixel sizes must be greater than zero."
        )

    ny, nx = screen.shape

    pixel_x = (
        nx // 2
        + np.rint(station_u_m / pixel_size_x_m).astype(int)
    )
    pixel_y = (
        ny // 2
        + np.rint(station_v_m / pixel_size_y_m).astype(int)
    )

    valid = (
        (pixel_x >= 0)
        & (pixel_x < nx)
        & (pixel_y >= 0)
        & (pixel_y < ny)
    )

    tec = np.zeros(station_u_m.shape, dtype=float)


    # NumPy indexes images as [y, x].
    tec[valid] = screen[
        pixel_y[valid],
        pixel_x[valid],
    ]

    return tec


def _tec_to_jones_gains(
    tec: np.ndarray,
    frequency_hz: np.ndarray,
) -> np.ndarray:
    """
    Convert station TEC values into diagonal Jones matrices.

    Parameters
    ----------
    tec
        TEC values with shape ``(time, antenna)`` in TECU.
    frequency_hz
        Gain frequencies with shape ``(frequency,)`` in Hz.

    Returns
    -------
    np.ndarray
        Gain matrices with shape
        ``(time, antenna, frequency, 2, 2)``.
    """
    tec = np.asarray(tec, dtype=float)
    frequency_hz = np.asarray(frequency_hz, dtype=float)

    if tec.ndim != 2:
        raise ValueError(
            "tec must have shape (time, antenna), "
            f"received {tec.shape}."
        )

    if frequency_hz.ndim != 1:
        raise ValueError(
            "frequency_hz must be one-dimensional, "
            f"received {frequency_hz.shape}."
        )

    if np.any(frequency_hz <= 0):
        raise ValueError("All gain frequencies must be positive.")

    phase = (
        -8.44797245e9
        * tec[:, :, np.newaxis]
        / frequency_hz[np.newaxis, np.newaxis, :]
    )

    scalar_gain = np.exp(1j * phase)

    gains = np.zeros(
        (*scalar_gain.shape, 2, 2),
        dtype=np.complex128,
    )

    gains[..., 0, 0] = scalar_gain
    gains[..., 1, 1] = scalar_gain

    # Cross-polarisation terms remain zero.
    return gains


def run_ionospheric_solver_ext(
    vis: Visibility,
    external_tec_fits: str,
    gaintable: GainTable,
) -> GainTable:
    """
    Construct phase-centre ionospheric gains from an external OSKAR TEC screen.

    This produces a direction-independent gain table for the observation phase
    centre. For the phase centre, the source direction cosines are ``l=m=0``;
    therefore each station samples the TEC screen using only its projected
    U/V position.

    The function uses the gain-table time and frequency grids. The returned
    gains have shape ``(time, antenna, frequency, receptor1, receptor2)``.

    Parameters
    ----------
    vis
        Visibility dataset containing the observation phase centre,
        configuration, and observation timing.
    external_tec_fits
        Path to the external FITS TEC-screen cube used by OSKAR.
    gaintable
        GainTable defining output solution times, frequencies, antennas,
        receptors, attributes, and chunking.

    Returns
    -------
    GainTable
        A copy of ``gaintable`` containing TEC-derived diagonal Jones gains,
        with the original chunking restored.
    """
    if gaintable.configuration is not None:
        configuration = gaintable.configuration
    elif vis.configuration is not None:
        configuration = vis.configuration
    else:
        raise ValueError(
            "A telescope configuration is required to obtain station "
            "coordinates."
        )

    if gaintable.phasecentre is not None:
        phasecentre = gaintable.phasecentre
    elif vis.phasecentre is not None:
        phasecentre = vis.phasecentre
    else:
        raise ValueError(
            "A phase centre is required for station UV projection."
        )

    station_xyz_m = np.asarray(configuration.xyz.data)

    if station_xyz_m.shape[0] != gaintable.sizes["antenna"]:
        raise ValueError(
            "Configuration and GainTable antenna counts differ: "
            f"{station_xyz_m.shape[0]} != "
            f"{gaintable.sizes['antenna']}."
        )

    location = configuration.location

    if location is None:
        raise ValueError(
            "The telescope configuration has no EarthLocation metadata."
        )

    gain_times = np.asarray(
        gaintable.coords["time"].data,
        dtype=float,
    )
    gain_frequencies = np.asarray(
        gaintable.coords["frequency"].data,
        dtype=float,
    )

    # SKA/RASCIL time coordinates are normally seconds since MJD = 0.
    gain_times_mjd = gain_times / 86400.0

    station_u_m, station_v_m = _project_stations_to_uvw(
        station_xyz_m=station_xyz_m,
        times_mjd=gain_times_mjd,
        phasecentre=phasecentre,
        longitude_rad=location.lon.rad,
    )

    original_chunks = gaintable.chunksizes

    with fits.open(
        external_tec_fits,
        mode="readonly",
        memmap=True,
    ) as hdul:
        hdu = hdul[0]
        header = hdu.header
        tec_cube = hdu.data

        # Remove only singleton axes while retaining TIME, Y and X.
        tec_cube = np.squeeze(tec_cube)

        if tec_cube.ndim == 2:
            # Static screen.
            tec_cube = tec_cube[np.newaxis, ...]
        elif tec_cube.ndim != 3:
            raise ValueError(
                "Expected TEC data with shape (time, y, x) or (y, x), "
                f"received {tec_cube.shape} after removing singleton axes."
            )

        pixel_size_x_m = abs(
            float(
                header.get(
                    "CDELT1",
                    header.get("CD1_1", 0.0),
                )
            )
        )
        pixel_size_y_m = abs(
            float(
                header.get(
                    "CDELT2",
                    header.get("CD2_2", 0.0),
                )
            )
        )

        if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
            raise ValueError(
                "Could not determine positive TEC-screen pixel sizes from "
                "CDELT1/CDELT2 or CD1_1/CD2_2."
            )

        screen_time_interval_s = abs(
            float(header.get("CDELT3", 0.0))
        )

        # OSKAR aligns screen zero with the observation start and ignores the
        # screen time-axis CRVAL/CRPIX.
        if "datetime" in vis:
            observation_start_datetime = np.asarray(
                vis["datetime"].data
            )[0]

            gain_datetimes = np.asarray(
                gaintable["datetime"].data
            )

            solution_elapsed_s = (
                gain_datetimes - observation_start_datetime
            ) / np.timedelta64(1, "s")

            # _screen_indices_for_times expects absolute-like values whose
            # difference gives elapsed seconds.
            screen_indices = _screen_indices_for_times(
                solution_times=solution_elapsed_s,
                observation_start_time=0.0,
                screen_time_interval_s=screen_time_interval_s,
                number_of_screens=tec_cube.shape[0],
            )
        else:
            vis_start_time = float(
                np.asarray(vis.coords["time"].data)[0]
            )

            screen_indices = _screen_indices_for_times(
                solution_times=gain_times,
                observation_start_time=vis_start_time,
                screen_time_interval_s=screen_time_interval_s,
                number_of_screens=tec_cube.shape[0],
            )

        tec_by_time_and_station = np.empty(
            (
                gain_times.size,
                station_xyz_m.shape[0],
            ),
            dtype=float,
        )

        cached_screen_index = None
        cached_screen = None

        for time_index, screen_index in enumerate(screen_indices):
            screen_index = int(screen_index)

            if screen_index != cached_screen_index:
                # With memmap enabled, only the requested screen plane is
                # accessed rather than copying the complete cube.
                cached_screen = tec_cube[screen_index]
                cached_screen_index = screen_index

            tec_by_time_and_station[time_index] = (
                _sample_phasecentre_tec(
                    screen=cached_screen,
                    station_u_m=station_u_m[time_index],
                    station_v_m=station_v_m[time_index],
                    pixel_size_x_m=pixel_size_x_m,
                    pixel_size_y_m=pixel_size_y_m,
                )
            )

    solved_gains = _tec_to_jones_gains(
        tec=tec_by_time_and_station,
        frequency_hz=gain_frequencies,
    )

    expected_shape = tuple(
        gaintable["gain"].shape
    )

    if solved_gains.shape != expected_shape:
        raise ValueError(
            "TEC-derived gains do not match the GainTable layout: "
            f"{solved_gains.shape} != {expected_shape}. "
            "Expected GainTable dimensions in the order "
            "(time, antenna, frequency, receptor1, receptor2)."
        )

    solved_gain_array = xr.DataArray(
        solved_gains,
        dims=gaintable["gain"].dims,
        coords=gaintable["gain"].coords,
        attrs=gaintable["gain"].attrs,
        name="gain",
    )

    LOGGER.info(
        "Created phase-centre TEC gains from %d screen planes for "
        "%d solution times, %d antennas and %d frequencies.",
        len(np.unique(screen_indices)),
        gain_times.size,
        station_xyz_m.shape[0],
        gain_frequencies.size,
    )

    solved_gaintable = gaintable.assign(
        gain=solved_gain_array,
    )

    if original_chunks:
        solved_gaintable = solved_gaintable.chunk(
            original_chunks
        )

    return solved_gaintable

def run_ionospheric_solver(
    vis: Visibility,
    modelvis: Visibility,
    gaintable: GainTable,
    cluster_indexes: np.ndarray = None,
    block_diagonal: bool = False,
    niter: int = 15,
    tol: float = 1e-6,
    zernike_limit: list[int] = None,
) -> GainTable:
    """
    Solve ionospheric phase screens for every gain-table solution interval.

    The solver uses the first visibility sample of each solution interval and
    the XX and YY correlations. Each gain-table chunk is solved independently
    through :meth:`xarray.DataArray.map_blocks`.

    Parameters
    ----------
    vis
        Observed visibility data. Its frequency and baseline coordinates must
        match those of ``modelvis``.
    modelvis
        Predicted visibility data corresponding to ``vis``.
    gaintable
        Gain table that provides solution intervals and initial gain values.
    cluster_indexes
        Integer cluster identifier for each antenna, with shape
        ``(n_antenna,)``. When omitted, all antennas belong to one cluster.
    block_diagonal
        Whether to solve the combined cluster system in one block-diagonal
        least-squares problem. Otherwise, clusters are solved sequentially.
    niter
        Maximum number of solver iterations per solution interval.
    tol
        Convergence threshold for the maximum fractional parameter change.
    zernike_limit
        Maximum Zernike order for each cluster. For cluster ``cid``, terms
        satisfy ``n + abs(m) <= zernike_limit[cid]``. The solver default is
        used when omitted.

    Returns
    -------
        A copy of ``gaintable`` with ionospheric gains, retaining its original
        chunking.
    """
    results_across_solints = []
    gaintable_time_coord = gaintable.coords["time"]
    gaintable_chunks = gaintable.chunksizes

    for idx, t_slice in enumerate(gaintable.soln_interval_slices):
        # Need to drop all time coords,
        # else map_block will not be able to align
        diagonal_vis_for_given_time = (
            vis.isel(time=t_slice)
            .isel(time=0, polarisation=[0, 3], drop=True)
            .chunk(-1)
        )
        diagonal_modelvis_for_given_time = (
            modelvis.isel(time=t_slice)
            .isel(time=0, polarisation=[0, 3], drop=True)
            .chunk(-1)
        )
        template_gaintable_gain = (
            gaintable["gain"].isel(time=idx, drop=True).chunk(-1)
        )

        res = xr.apply_ufunc(
            _run_ionospheric_solver_ufunc_,
            template_gaintable_gain,
            diagonal_vis_for_given_time["vis"],
            diagonal_vis_for_given_time["weight"],
            diagonal_vis_for_given_time["flags"],
            diagonal_modelvis_for_given_time["vis"],
            input_core_dims=[
                ["antenna", "frequency", "receptor1", "receptor2"],
                ["baselineid", "frequency", "polarisation"],
                ["baselineid", "frequency", "polarisation"],
                ["baselineid", "frequency", "polarisation"],
                ["baselineid", "frequency", "polarisation"],
            ],
            output_core_dims=[
                ["antenna", "frequency", "receptor1", "receptor2"]
            ],
            kwargs=dict(
                antenna1=vis["antenna1"].values,
                antenna2=vis["antenna2"].values,
                frequency=vis["frequency"].values,
                configuration=vis.configuration,
                cluster_indexes=cluster_indexes,
                block_diagonal=block_diagonal,
                niter=niter,
                tol=tol,
                zernike_limit=zernike_limit,
            ),
            dask="parallelized",
            output_dtypes=[template_gaintable_gain.dtype],
        )

        results_across_solints.append(res)

    concat_gaintable_gain = xr.concat(
        results_across_solints,
        dim=gaintable_time_coord,
    )

    return gaintable.assign(gain=concat_gaintable_gain).chunk(gaintable_chunks)


def _run_ionospheric_solver_ufunc_(
    gaintable_gain: np.ndarray,
    vis_vis: np.ndarray,
    vis_weight: np.ndarray,
    vis_flags: np.ndarray,
    modelvis_vis: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    frequency: np.ndarray,
    configuration: Configuration,
    cluster_indexes: np.ndarray = None,
    block_diagonal: bool = False,
    niter: int = 15,
    tol: float = 1e-6,
    zernike_limit: list[int] = None,
):
    """Solve one set of xarray core dimensions with NumPy inputs."""
    gain = IonosphericSolver(
        vis_vis,
        vis_weight,
        vis_flags,
        modelvis_vis,
        antenna1,
        antenna2,
        frequency,
        configuration,
        cluster_indexes,
        block_diagonal,
        niter,
        tol,
        zernike_limit,
    ).solve(gaintable_gain)

    return gain


class IonosphericSolver:
    """
    Solve ionospheric phase screens using a linearized approach.

    This class sets up and solves a system of linear equations to determine
    the parameters of a phase screen model (e.g., Zernike polynomials) that
    best fits the observed visibility data. It supports antenna clustering
    and iterative refinement of the solution.

    The visibility inputs represent the zeroth time sample of one gain-table
    solution interval. They must contain exactly two linear-polarisation
    correlations, ordered as XX and YY.

    Parameters
    ----------
    vis_vis
        Complex-valued observed visibility data for the zeroth time sample of
        one solution interval, with shape
        ``(n_baseline, n_frequency, 2)``. The final dimension must contain,
        in order, the XX and YY correlations only.
    vis_weight
        Real-valued visibility weights for ``vis_vis``, with the same shape.
    vis_flags
        Boolean visibility flags for ``vis_vis``, with the same shape. Flagged
        samples have zero weight in the normal equations.
    modelvis_vis
        Complex-valued model visibility data for the same time sample and
        correlations as ``vis_vis``, with the same shape.
    antenna1
        Integer first-antenna index for each baseline, with shape
        ``(n_baseline,)``.
    antenna2
        Integer second-antenna index for each baseline, with shape
        ``(n_baseline,)``.
    frequency
        Real-valued channel frequencies in Hz, with shape ``(n_frequency,)``.
    configuration
        Telescope configuration
    cluster_indexes
        Integer cluster identifiers with shape ``(n_antenna,)``. Identifiers
        must be contiguous from zero. If omitted, all antennas share cluster
        zero.
    block_diagonal
        If True, solve for all clusters simultaneously assuming a
        block-diagonal system. If False, solve for each cluster sequentially
        (default: False).
    niter
        Maximum number of iterations for the solver.
    tol
        Tolerance for the fractional change in parameters to determine
        convergence
    zernike_limit
        Maximum Zernike order for each cluster. For cluster ``cid``, terms
        satisfy ``n + abs(m) <= zernike_limit[cid]``. The solver default is
        used when omitted.

    Raises
    ------
    ValueError
        If model visibilities are all zero or if length of ``cluster_indexes``
        does not match number of antennas.
    """

    def __init__(
        self,
        vis_vis: np.ndarray,
        vis_weight: np.ndarray,
        vis_flags: np.ndarray,
        modelvis_vis: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        frequency: np.ndarray,
        configuration: Configuration,
        cluster_indexes: np.ndarray = None,
        block_diagonal: bool = False,
        niter: int = 15,
        tol: float = 1e-6,
        zernike_limit: list[int] = None,
    ):
        if np.all(modelvis_vis == 0.0):
            raise ValueError("solve_ionosphere: Model visibilities are zero")

        self.change = np.inf

        self.cluster_indexes = cluster_indexes
        self.block_diagonal = block_diagonal
        self.niter = niter
        self.tol = tol
        self.zernike_limit = zernike_limit

        self.vis = vis_vis
        self.weight = vis_weight
        self.flags = vis_flags
        self.modelvis = modelvis_vis

        self.xyz = configuration.xyz
        self.antenna1 = antenna1
        self.antenna2 = antenna2
        # Cross-corelation mask
        self.mask0 = self.antenna1 != self.antenna2

        self.wl_const = (
            2.0 * np.pi * const.c.value / frequency  # pylint: disable=E1101
        )

        n_antenna = configuration.sizes["id"]
        if self.cluster_indexes is None:
            self.cluster_indexes = np.zeros(n_antenna, np.int32)

        if n_antenna != len(self.cluster_indexes):
            raise ValueError(
                f"cluster_indexes has wrong size {len(self.cluster_indexes)}"
            )

        self.param, self.coeff = set_coeffs_and_params(
            self.xyz, self.cluster_indexes, self.zernike_limit
        )
        # Need to convert list to numpy array
        self.param = np.asarray(self.param)

        n_cluster = np.max(self.cluster_indexes) + 1
        n_param = get_param_count(self.param)[0]
        if n_cluster == 1:
            logger.info(
                "Setting up iono solver for %d stations in a single cluster",
                n_antenna,
            )
            logger.info(
                "There are %d total parameters in the cluster", n_param
            )
        else:
            logger.info(
                "Setting up iono solver for %d stations in %d clusters",
                n_antenna,
                n_cluster,
            )
            logger.info(
                "There are %d total parameters: %d in c[0] + %d x c[1:%d]",
                n_param,
                len(self.param[0]),
                len(self.param[1]),
                len(self.param) - 1,
            )

    def solve(self, gaintable_gain: np.ndarray):
        """
        Solve the ionospheric phase screen for single
        gain-table solution interval.

        Parameters
        ----------
        gaintable_gain
            Initial gain values for the solution interval, with shape
            ``(n_antenna, n_frequency, 2, 2)``.

        Returns
        -------
        numpy.ndarray
            Ionospheric gains for the same solution interval, with shape
            ``(n_antenna, n_frequency, 2, 2)``.
        """
        param = self._get_updated_params()
        new_gain_data = self._update_gain_table(param, gaintable_gain)

        return new_gain_data

    def _get_updated_params(self) -> np.ndarray:
        """
        Iteratively update the screen parameters until convergence.

        In each iteration, this method calls the selected solver function to
        get a parameter update, adds the update to the current parameters,
        and applies the resulting phase distortion to the model visibilities
        for the next iteration.

        Returns
        -------
            The final, converged screen parameters.
        """
        modelvis = self.modelvis
        param = self.param
        solve_function = (
            self._solve_for_block_diagonal
            if self.block_diagonal
            else self._solve_for_non_block_diagonal
        )

        for it in range(self.niter):
            param_update = solve_function(modelvis, param, it)
            param = param_update + param
            modelvis = self._apply_phase_distortions(modelvis, param_update)

        return param

    def _build_normal_equation(
        self,
        modelvis: np.ndarray,
        param: np.ndarray,
        cid: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct the normal equation matrices AA and Ab.

        This function builds the linear system AA . x = Ab, where
        AA = Real(A^H W A) and Ab = Imag(A^H W dV).

        Parameters
        ----------
        modelvis
            The model visibilities.
        param
            The current screen parameters.
        cid
            The cluster ID. If specified, the equation is built only for this
            cluster. Otherwise, it's built for all clusters (default: None).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the matrices AA and Ab.
        """
        A = self._build_cluster_design_matrix(modelvis, param, cid)

        n_param, *_ = A.shape

        AA = np.zeros((n_param, n_param))
        Ab = np.zeros(n_param)

        A_sliced = A
        wgt = self.weight * (1 - self.flags)
        vis_diff = self.vis - modelvis

        AA = np.real(
            np.einsum(
                "pbfo,bfo,qbfo->pq",
                np.conj(A_sliced),
                wgt,
                A_sliced,
                optimize=True,
            )
        )

        Ab = np.imag(
            np.einsum(
                "pbfo,bfo,bfo->p",
                np.conj(A_sliced),
                wgt,
                vis_diff,
                optimize=True,
            )
        )

        return AA, Ab

    def _build_cluster_design_matrix(
        self,
        modelvis,
        param,
        cid=None,
    ):
        """
        Build the design matrix A for the linear system.

        The design matrix A relates the visibility phases to the screen
        parameters. This method constructs A for either a single specified
        cluster or for all clusters combined.

        Parameters
        ----------
        modelvis : numpy.ndarray
            The model visibilities.
        param : numpy.ndarray
            The current screen parameters.
        cid : int, optional
            Cluster ID. If given, the matrix is built only for this cluster.
            Otherwise, the matrix for all clusters is returned

        Returns
        -------
        numpy.ndarray
            The complex-valued design matrix A of shape
            (n_param, n_baselines, n_freq, n_pol).
        """
        [n_cluster, _, stn2cid] = set_cluster_maps(self.cluster_indexes)

        if cid is not None:
            return self._cluster_design_matrix(
                modelvis,
                stn2cid,
                len(param[cid]),
                cid,
            )

        [n_param, pidx0] = get_param_count(param)

        A = np.zeros((n_param, *modelvis.shape), np.complex128)

        for _cid in range(0, n_cluster):
            pid = np.arange(pidx0[_cid], pidx0[_cid] + len(param[_cid]))

            A[pid, :] += self._cluster_design_matrix(
                modelvis,
                stn2cid,
                len(param[_cid]),
                _cid,
            )

        return A

    def _cluster_design_matrix(
        self,
        modelvis,
        stn2cid,
        n_param,
        cid,
    ):
        """
        Calculate the design matrix for a single, specified cluster.

        This is a helper method that computes the components of the design
        matrix A for a specific cluster.

        Parameters
        ----------
        modelvis : numpy.ndarray
            The model visibilities.
        stn2cid : list of list
            Mapping from station ID to cluster ID.
        n_param : int
            The number of parameters for this cluster.
        cid : int
            The ID of the cluster for which to build the matrix.

        Returns
        -------
        numpy.ndarray
            The design matrix A for the specified cluster.
        """
        n_baselines = len(self.mask0)
        A = np.zeros((n_param, *modelvis.shape), np.complex128)
        wl_const = self.wl_const.reshape(1, *self.wl_const.shape, 1)

        blidx_all = np.arange(n_baselines)
        blidx = blidx_all[self.mask0 * (stn2cid[self.antenna1] == cid)]

        if len(blidx) > 0:
            A[:, blidx, :] += np.einsum(
                "bfq,bp->pbfq",
                modelvis[blidx, :, :] * wl_const,
                np.vstack(self.coeff[self.antenna1[blidx]]).astype(np.float32),
            )

        blidx = blidx_all[self.mask0 * (stn2cid[self.antenna2] == cid)]
        if len(blidx) > 0:
            A[:, blidx, :] -= np.einsum(
                "bfq,bp->pbfq",
                modelvis[blidx, :, :] * wl_const,
                np.vstack(self.coeff[self.antenna2[blidx]]).astype(np.float32),
            )

        return A

    def _update_gain_table(
        self, param: np.ndarray, gain_data: np.ndarray
    ) -> np.ndarray:
        """
        Construct the final gain table from the solved parameters.

        This method uses the final screen parameters to compute the complex
        gains for each antenna and frequency.

        Parameters
        ----------
        param
            The final, converged screen parameters for all clusters.
        gain_data
            An empty or template gain data array to be filled.
            Shape: (antenna, freq, 2, 2)

        Returns
        -------
            The populated gain table data array. Same shape as gain_data
        """
        [n_cluster, cid2stn, _] = set_cluster_maps(self.cluster_indexes)
        table_data = np.copy(gain_data)

        for cid in range(0, n_cluster):
            # combine parmas for [n_station] phase terms and scale for [n_freq]
            diag_gain = np.exp(
                np.einsum(
                    "s,f->sf",
                    np.einsum(
                        "sp,p->s",
                        np.vstack(self.coeff[cid2stn[cid]]).astype(np.float64),
                        param[cid],
                    ),
                    1j * self.wl_const,
                )
            )

            table_data[cid2stn[cid], :, 0, 0] = diag_gain
            table_data[cid2stn[cid], :, 1, 1] = diag_gain

        return table_data

    def _solve_for_block_diagonal(
        self, modelvis: np.ndarray, param: np.ndarray, it: int
    ):
        """
        Solve the normal equation for all clusters at once (block-diagonal).

        This method assumes the system matrix is block-diagonal and solves
        for all parameters of all clusters in a single least-squares problem.

        Parameters
        ----------
        modelvis
            The model visibilities, possibly updated from a previous iteration.
        param
            The current screen parameters for all clusters.
        it
            The current iteration number.

        Returns
        -------
        numpy.ndarray
            The calculated parameter update for this iteration.
        """
        n_cluster = np.max(self.cluster_indexes) + 1
        param_update = np.zeros((n_cluster, param.shape[-1]), param.dtype)

        if self.change < self.tol:
            return param_update

        [_, pidx0] = get_param_count(param)

        [AA, Ab] = self._build_normal_equation(modelvis, param)
        soln_vec = np.linalg.lstsq(AA, Ab, rcond=None)[0]

        nu = 1.0 - 0.5 * (it % 2)
        for cid in range(n_cluster):
            param_update[cid] = (
                nu
                * soln_vec[
                    pidx0[cid] : pidx0[cid] + len(param[cid])  # noqa:E203
                ]
            )

        self._update_and_log_change(param_update, param + param_update, it)
        return param_update

    def _solve_for_non_block_diagonal(self, modelvis, param, it):
        """
        Solve the normal equation for each cluster sequentially.

        This method iterates through each cluster, building and solving a
        separate least-squares problem for each one.

        Parameters
        ----------
        modelvis : dask.array.Array or numpy.ndarray
            The model visibilities, possibly updated from a previous iteration.
        param : dask.array.Array or numpy.ndarray
            The current screen parameters for all clusters.
        it : int
            The current iteration number.

        Returns
        -------
        numpy.ndarray
            The calculated parameter update for this iteration.
        """
        n_cluster = np.max(self.cluster_indexes) + 1
        param_update = np.zeros((n_cluster, param.shape[-1]), param.dtype)

        if self.change < self.tol:
            return param_update

        for cid in range(n_cluster):
            [AA, Ab] = self._build_normal_equation(modelvis, param, cid)

            # Solve the current incremental normal equations
            soln_vec = np.linalg.lstsq(AA, Ab, rcond=None)[0]

            # Update factor
            nu = 0.5
            # nu = 1.0 - 0.5 * (it % 2)
            param_update[cid] = nu * soln_vec

        self._update_and_log_change(param_update, param + param_update, it)
        return param_update

    def _apply_phase_distortions(self, vis: np.ndarray, param: np.ndarray):
        """
        Apply solved phase distortions to visibilities.

        This method uses a set of screen parameters to calculate the
        corresponding phase screen and applies it to the input visibilities.

        Parameters
        ----------
        vis
            The visibilities to which the phase distortions will be applied.
            Shape: (baseline, freq, pol)
        param
            The screen parameters for all clusters.
            Shape: (n_clusters, n_params)

        Returns
        -------
        numpy.ndarray
            The visibilities with the phase distortions applied.
        """
        if self.change < self.tol:
            return vis

        vis = np.copy(vis)
        [n_cluster, _, stn2cid] = set_cluster_maps(self.cluster_indexes)

        for cid1, cid2 in np.ndindex(n_cluster, n_cluster):
            # A mask for all baselines in this cluster pair
            mask = (
                self.mask0
                * (stn2cid[self.antenna1] == cid1)
                * (stn2cid[self.antenna2] == cid2)
            )
            if np.sum(mask) == 0:
                continue

            coeffs1 = np.vstack(self.coeff[self.antenna1[mask]]).astype(
                np.float64
            )
            coeffs2 = np.vstack(self.coeff[self.antenna2[mask]]).astype(
                np.float64
            )

            tec_effect1 = np.einsum("bp,p->b", coeffs1, param[cid1])
            tec_effect2 = np.einsum("bp,p->b", coeffs2, param[cid2])
            baseline_tec_diff = tec_effect1 - tec_effect2

            baseline_phase = np.einsum(
                "b,f->bf", baseline_tec_diff, 1j * self.wl_const
            )

            vis[mask, :, :] *= np.exp(baseline_phase)[..., np.newaxis]

        return vis

    def _update_and_log_change(self, param_update, param, it):
        """
        Calculate and log the fractional change in parameters.

        This method computes the maximum fractional change between the
        parameter update and the new parameter values to monitor convergence.
        The result is stored in `self.change` and logged.

        Parameters
        ----------
        param_update : numpy.ndarray
            The parameter updates from the latest solver iteration.
        param : numpy.ndarray
            The newly updated parameters (current + update).
        it : int
            The current iteration number.
        """
        eps = 1e-13

        self.change = np.max(
            np.abs(np.hstack(param_update).astype(np.float64))
            / np.abs(np.hstack(param + eps).astype(np.float64))
        )

        logger.info(
            "Ionospheric Solver: Iteration %d, change: %f", it, self.change
        )
