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


def run_ionospheric_solver(
    vis: Visibility,
    modelvis: Visibility,
    gaintable: GainTable,
    cluster_indexes: np.ndarray = None,
    block_diagonal: bool = False,
    niter: int = 15,
    tol: float = 1e-6,
    zernike_limit: list[int] = None,
):
    results_across_solints = []
    gaintable_time_coord = gaintable.coords["time"]
    gaintable_chunks = gaintable.chunksizes

    for idx, t_slice in enumerate(gaintable.soln_interval_slices):
        # Need to drop all time coords, else map_block will not be able to align
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

        res = template_gaintable_gain.map_blocks(
            _run_ionospheric_solver_block_,
            args=(
                diagonal_vis_for_given_time["vis"],
                diagonal_vis_for_given_time["weight"],
                diagonal_vis_for_given_time["flags"],
                diagonal_modelvis_for_given_time["vis"],
            ),
            kwargs=dict(
                antenna1=vis["antenna1"],
                antenna2=vis["antenna2"],
                frequency=vis["frequency"],
                configuration=vis.configuration,
                cluster_indexes=cluster_indexes,
                block_diagonal=block_diagonal,
                niter=niter,
                tol=tol,
                zernike_limit=zernike_limit,
            ),
            template=template_gaintable_gain,
        )

        results_across_solints.append(res)

    concat_gaintable_gain = xr.concat(
        results_across_solints,
        dim=gaintable_time_coord,
    )

    return gaintable.assign(gain=concat_gaintable_gain).chunk(gaintable_chunks)


def _run_ionospheric_solver_block_(
    gaintable_gain: xr.DataArray,
    vis_vis: xr.DataArray,
    vis_weight: xr.DataArray,
    vis_flags: xr.DataArray,
    modelvis_vis: xr.DataArray,
    antenna1: xr.DataArray,
    antenna2: xr.DataArray,
    frequency: xr.DataArray,
    configuration: Configuration,
    cluster_indexes: np.ndarray = None,
    block_diagonal: bool = False,
    niter: int = 15,
    tol: float = 1e-6,
    zernike_limit: list[int] = None,
):
    gain = IonosphericSolver(
        vis_vis.values,
        vis_weight.values,
        vis_flags.values,
        modelvis_vis.values,
        antenna1.values,
        antenna2.values,
        frequency.values,
        configuration,
        cluster_indexes,
        block_diagonal,
        niter,
        tol,
        zernike_limit,
    )._solve(gaintable_gain.values)

    new_gain_xdr = gaintable_gain.copy()
    new_gain_xdr.data = gain

    return new_gain_xdr


class IonosphericSolver:
    """
    Solves for ionospheric phase screens using a linearized approach.

    This class sets up and solves a system of linear equations to determine
    the parameters of a phase screen model (e.g., Zernike polynomials) that
    best fits the observed visibility data. It supports antenna clustering
    and iterative refinement of the solution.

    NOTE: This solver assumes that input visibilities have linear
    polarisation, with all 4 polarisations (XX, XY, YX, YY) present.

    Parameters
    ----------
    vis_vis: np.ndarray,
    vis_weight: np.ndarray,
    vis_flags: np.ndarray,
    modelvis_vis: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    frequency: np.ndarray,
    configuration: Configuration,
    cluster_indexes
        Array of integers assigning each antenna to a cluster. If None, all
        antennas are treated as a single cluster (default: None).
    block_diagonal
        If True, solve for all clusters simultaneously assuming a
        block-diagonal system. If False, solve for each cluster sequentially
        (default: False).
    niter
        Maximum number of iterations for the solver (default: 15).
    tol
        Tolerance for the fractional change in parameters to determine
        convergence (default: 1e-6).
    zernike_limit
        list of Zernike index limits:
        Generate all Zernikes with n + |m| <= zernike_limit[cluster_id].
        If None, a default is used by the solver.

    Attributes
    ----------
    xyz : numpy.ndarray
        Cartesian coordinates of the antennas.
    antenna1 : numpy.ndarray
        Array of first antenna indices for each baseline.
    antenna2 : numpy.ndarray
        Array of second antenna indices for each baseline.
    change : float
        The maximum fractional change in parameters from the last iteration.
    niter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    vis : xarray.DataArray
        Observed visibilities for the first time step.
    weight : xarray.DataArray
        Visibility weights for the first time step.
    flags : xarray.DataArray
        Visibility flags for the first time step.
    modelvis : xarray.DataArray
        Model visibilities for the first time step.
    pols : list
        List of polarization indices to be used in the solve.
    param : list of numpy.ndarray
        List of parameter arrays, one for each cluster.
    coeff : list of numpy.ndarray
        List of coefficient arrays, one for each antenna.
    solve_function : callable
        The method used to solve the normal equations, chosen based on
        `block_diagonal`.

    Raises
    ------
    ValueError
        If model visibilities are all zero or if polarisations are unsupported.
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

    def _solve(self, gaintable_gain: np.ndarray):
        """
        Execute the ionospheric phase screen solver.

        Parameters
        ----------
        gaintable_gain
            Initial gain values
            Shape: (antenna, freq, rec1, rec2)

        Returns
        -------
            New gain data
            Shape: (antenna, freq, rec1, rec2)
        """
        param = self.get_updated_params()
        new_gain_data = self.updated_gain_table(param, gaintable_gain)

        return new_gain_data

    def get_updated_params(self):
        """
        Iteratively update the screen parameters until convergence.

        In each iteration, this method calls the selected solver function to
        get a parameter update, adds the update to the current parameters,
        and applies the resulting phase distortion to the model visibilities
        for the next iteration.

        Parameters
        ----------
        n_cluster : int
            The number of antenna clusters.
        n_param : int
            The total number of parameters to solve for.

        Returns
        -------
        dask.array.Array
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

    def build_normal_equation(
        self,
        modelvis: np.ndarray,
        param: np.ndarray,
        cid: int | None = None,
    ):
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
        A = self.build_cluster_design_matrix(modelvis, param, cid)

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

    def build_cluster_design_matrix(
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
            Otherwise, the matrix for all clusters is returned (default: None).

        Returns
        -------
        numpy.ndarray
            The complex-valued design matrix A of shape
            (n_param, n_baselines, n_freq, n_pol).
        """
        [n_cluster, _, stn2cid] = set_cluster_maps(self.cluster_indexes)

        if cid is not None:
            return self.cluster_design_matrix(
                modelvis,
                stn2cid,
                len(param[cid]),
                cid,
            )

        [n_param, pidx0] = get_param_count(param)

        A = np.zeros((n_param, *modelvis.shape), np.complex128)

        for _cid in range(0, n_cluster):
            pid = np.arange(pidx0[_cid], pidx0[_cid] + len(param[_cid]))

            A[pid, :] += self.cluster_design_matrix(
                modelvis,
                stn2cid,
                len(param[_cid]),
                _cid,
            )

        return A

    def cluster_design_matrix(
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

    def updated_gain_table(
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
            Shape: (antenna, freq, rec1, rec2)

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

        [AA, Ab] = self.build_normal_equation(modelvis, param)
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
            [AA, Ab] = self.build_normal_equation(modelvis, param, cid)

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
