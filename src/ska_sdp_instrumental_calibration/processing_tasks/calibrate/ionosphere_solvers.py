import logging

import dask
import dask.array as da
import numpy as np
import xarray as xr
from astropy import constants as const
from ska_sdp_func_python.calibration.ionosphere_solvers import (
    get_param_count,
    set_cluster_maps,
    set_coeffs_and_params,
)

from ska_sdp_instrumental_calibration.workflow.utils import (
    create_bandpass_table,
    create_grouped_array,
    get_indices_from_grouped_bins,
)

log = logging.getLogger("func-python-logger")


class IonosphericSolver:
    """
    Solves for ionospheric phase screens using a linearized approach.

    This class sets up and solves a system of linear equations to determine
    the parameters of a phase screen model (e.g., Zernike polynomials) that
    best fits the observed visibility data. It supports antenna clustering
    and iterative refinement of the solution.

    Parameters
    ----------
    vis : xarray.Dataset
        Input visibility dataset.
    modelvis : xarray.Dataset
        Model visibility dataset corresponding to `vis`.
    cluster_indexes : numpy.ndarray, optional
        Array of integers assigning each antenna to a cluster. If None, all
        antennas are treated as a single cluster (default: None).
    block_diagonal : bool, optional
        If True, solve for all clusters simultaneously assuming a
        block-diagonal system. If False, solve for each cluster sequentially
        (default: False).
    niter : int, optional
        Maximum number of iterations for the solver (default: 15).
    tol : float, optional
        Tolerance for the fractional change in parameters to determine
        convergence (default: 1e-6).
    zernike_limit : int, optional
        The maximum order of Zernike polynomials to use for the screen model.
        If None, a default is used (default: None).

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

    @staticmethod
    def solve_calibrator(
        vis,
        modelvis,
        cluster_indexes=None,
        block_diagonal=False,
        niter=15,
        tol=1e-6,
        zernike_limit=None,
    ):
        gaintable = create_bandpass_table(vis)
        solver = IonosphericSolver(
            vis,
            modelvis,
            cluster_indexes,
            block_diagonal,
            niter,
            tol,
            zernike_limit,
        )
        return solver.solve(gaintable)

    @staticmethod
    def solve_target(
        vis,
        modelvis,
        timeslice,
        cluster_indexes=None,
        block_diagonal=False,
        niter=15,
        tol=1e-6,
        zernike_limit=None,
    ):

        return xr.concat(
            [
                IonosphericSolver.solve_calibrator(
                    vis.isel(time=t_slice),
                    modelvis.isel(time=t_slice),
                    cluster_indexes,
                    block_diagonal,
                    niter,
                    tol,
                    zernike_limit,
                )
                for t_slice in get_indices_from_grouped_bins(
                    create_grouped_array(vis.time, "time", timeslice)
                )
            ],
            dim="time",
        )

    def __init__(
        self,
        vis,
        modelvis,
        cluster_indexes=None,
        block_diagonal=False,
        niter=15,
        tol=1e-6,
        zernike_limit=None,
    ):

        if np.all(modelvis.vis == 0.0):
            raise ValueError("solve_ionosphere: Model visibilities are zero")

        self.pols = [0]

        if vis.visibility_acc.polarisation_frame.type.find("stokesI") == 0:
            self.pols = np.array(
                [np.argwhere(vis.polarisation.data == "I")[0][0]]
            )
        elif vis.visibility_acc.polarisation_frame.type.find("linear") == 0:
            self.pols = np.array(
                [
                    np.argwhere(vis.polarisation.data == "XX")[0][0],
                    np.argwhere(vis.polarisation.data == "YY")[0][0],
                ]
            )
        else:
            raise ValueError(
                "build_normal_equation: Unsupported polarisations"
            )

        self.change = np.inf

        self.cluster_indexes = cluster_indexes
        self.block_diagonal = block_diagonal
        self.niter = niter
        self.tol = tol
        self.zernike_limit = zernike_limit

        self.vis = vis.vis.isel(time=0)
        self.weight = vis.weight.isel(time=0)
        self.flags = vis.flags.isel(time=0)
        self.modelvis = modelvis.vis.isel(time=0)

        self.xyz = vis.configuration.xyz
        self.antenna1 = vis.antenna1.data
        self.antenna2 = vis.antenna2.data
        self.mask0 = self.antenna1 != self.antenna2

        self.wl_const = (
            2.0
            * np.pi
            * const.c.value  # pylint: disable=E1101
            / vis.frequency.data
        )

    def solve(self, gaintable=None):
        """
        Execute the ionospheric phase screen solver.

        This method orchestrates the solving process. It initializes the
        parameters and coefficients, iteratively updates them by solving the
        normal equations, and finally constructs a gain table representing
        the solved phase screen.

        Parameters
        ----------
        gaintable: xarray.Dataset
            Input gaintable. Default None

        Returns
        -------
        xarray.Dataset
            A gain table containing the solved ionospheric phase corrections.

        Raises
        ------
        ValueError
            If the `cluster_indexes` array has an incorrect size.
        """
        if self.cluster_indexes is None:
            self.cluster_indexes = np.zeros(len(gaintable.antenna), "int")

        if len(gaintable.antenna) != len(self.cluster_indexes):
            raise ValueError(
                f"cluster_indexes has wrong size {len(self.cluster_indexes)}"
            )

        [self.param, self.coeff] = set_coeffs_and_params(
            self.xyz, self.cluster_indexes, self.zernike_limit
        )

        n_cluster = np.amax(self.cluster_indexes) + 1
        n_param = get_param_count(self.param)[0]
        if n_cluster == 1:
            log.info(
                "Setting up iono solver for %d stations in a single cluster",
                len(gaintable.antenna),
            )
            log.info("There are %d total parameters in the cluster", n_param)
        else:
            log.info(
                "Setting up iono solver for %d stations in %d clusters",
                len(gaintable.antenna),
                n_cluster,
            )
            log.info(
                "There are %d total parameters: %d in c[0] + %d x c[1:%d]",
                n_param,
                len(self.param[0]),
                len(self.param[1]),
                len(self.param) - 1,
            )

        param = self.get_updated_params(n_cluster, n_param)
        new_gain_data = da.from_delayed(
            self.updated_gain_table(param, gaintable.gain),
            gaintable.gain.shape,
            gaintable.gain.dtype,
        )

        new_gain = gaintable.gain.copy()
        new_gain.data = new_gain_data

        return gaintable.assign({"gain": new_gain})

    def get_updated_params(self, n_cluster, n_param):
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
        param = da.from_array(self.param, chunks="auto")
        solve_function = dask.delayed(
            self._solve_for_block_diagonal
            if self.block_diagonal
            else self._solve_for_non_block_diagonal
        )

        for it in range(self.niter):
            param_update = da.from_delayed(
                solve_function(modelvis, param, it),
                (n_cluster, n_param),
                np.float64,
            )

            param = param_update + param

            modelvis = da.from_delayed(
                dask.delayed(self._apply_phase_distortions)(
                    modelvis, param_update
                ),
                modelvis.shape,
                modelvis.dtype,
            )

        return param

    def build_normal_equation(
        self,
        modelvis,
        param,
        cid=None,
    ):
        """
        Construct the normal equation matrices AA and Ab.

        This function builds the linear system AA . x = Ab, where
        AA = Real(A^H W A) and Ab = Imag(A^H W dV).

        Parameters
        ----------
        modelvis : numpy.ndarray
            The model visibilities.
        param : numpy.ndarray
            The current screen parameters.
        cid : int, optional
            The cluster ID. If specified, the equation is built only for this
            cluster. Otherwise, it's built for all clusters (default: None).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the matrices AA and Ab.
        """
        A = self.build_cluster_design_matrix(modelvis, param, cid)

        (n_param, *_) = A.shape

        AA = np.zeros((n_param, n_param))
        Ab = np.zeros(n_param)

        A_sliced = A[..., self.pols]
        wgt = self.weight[..., self.pols] * (1 - self.flags[..., self.pols])
        vis_diff = self.vis[..., self.pols] - modelvis[..., self.pols]

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

        A = np.zeros((n_param, *modelvis.shape), "complex_")

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
        A = np.zeros((n_param, *modelvis.shape), "complex_")
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

    @dask.delayed
    def updated_gain_table(self, param, gain):
        """
        Construct the final gain table from the solved parameters.

        This method uses the final screen parameters to compute the complex
        gains for each antenna and frequency.

        Parameters
        ----------
        param : numpy.ndarray
            The final, converged screen parameters for all clusters.
        gain : numpy.ndarray
            An empty or template gain data array to be filled.

        Returns
        -------
        numpy.ndarray
            The populated gain table data array.
        """
        [n_cluster, cid2stn, _] = set_cluster_maps(self.cluster_indexes)
        table_data = np.copy(gain.data)

        for cid in range(0, n_cluster):
            # combine parmas for [n_station] phase terms and scale for [n_freq]
            table_data[0, cid2stn[cid], :, 0, 0] = np.exp(
                np.einsum(
                    "s,f->sf",
                    np.einsum(
                        "sp,p->s",
                        np.vstack(self.coeff[cid2stn[cid]]).astype("float_"),
                        param[cid],
                    ),
                    1j * self.wl_const,
                )
            )

        return table_data

    def _solve_for_block_diagonal(self, modelvis, param, it):
        """
        Solve the normal equation for all clusters at once (block-diagonal).

        This method assumes the system matrix is block-diagonal and solves
        for all parameters of all clusters in a single least-squares problem.

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
        n_cluster = np.amax(self.cluster_indexes) + 1
        param_update = np.zeros((n_cluster, param.shape[-1]), param.dtype)

        if self.change < self.tol:
            return param_update

        [_, pidx0] = get_param_count(param)

        [AA, Ab] = self.build_normal_equation(modelvis, param)
        soln_vec = np.linalg.lstsq(AA, Ab, rcond=None)[0]

        nu = 0.5
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
        n_cluster = np.amax(self.cluster_indexes) + 1
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

    def _apply_phase_distortions(self, vis, param):
        """
        Apply solved phase distortions to visibilities.

        This method uses a set of screen parameters to calculate the
        corresponding phase screen and applies it to the input visibilities.

        Parameters
        ----------
        vis : numpy.ndarray
            The visibilities to which the phase distortions will be applied.
        param : numpy.ndarray
            The screen parameters for all clusters.

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
                "float_"
            )
            coeffs2 = np.vstack(self.coeff[self.antenna2[mask]]).astype(
                "float_"
            )

            tec_effect1 = np.einsum("bp,p->b", coeffs1, param[cid1])
            tec_effect2 = np.einsum("bp,p->b", coeffs2, param[cid2])
            baseline_tec_diff = tec_effect1 - tec_effect2

            baseline_phase = np.einsum(
                "b,f->bf", baseline_tec_diff, 1j * self.wl_const
            )

            vis[mask, :, 0] *= np.exp(baseline_phase)

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
        mask = np.abs(np.hstack(param).astype("float_")) > 0.0
        self.change = np.max(
            np.abs(np.hstack(param_update)[mask].astype("float_"))
            / np.abs(np.hstack(param)[mask].astype("float_"))
        )

        log.info(
            "Ionospheric Solver: Iteration %d, change: %f", it, self.change
        )
