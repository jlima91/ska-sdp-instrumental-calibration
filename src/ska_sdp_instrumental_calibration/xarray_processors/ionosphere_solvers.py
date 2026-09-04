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
