# flake8: noqa
# pylint: skip-file
import logging

import dask
import dask.array as da
import numpy as np
from astropy import constants as const
from ska_sdp_func_python.calibration.ionosphere_solvers import (
    get_param_count,
    set_cluster_maps,
    set_coeffs_and_params,
    solve_normal_equation,
)

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    restore_baselines_dim,
)
from ska_sdp_instrumental_calibration.workflow.utils import (
    create_bandpass_table,
    with_chunks,
)

log = logging.getLogger("func-python-logger")


class IonosphericSolver:
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

        self.xyz = vis.configuration.xyz
        self.antenna1 = vis.antenna1.data
        self.antenna2 = vis.antenna2.data
        self.change = np.inf

        self.niter = niter
        self.tol = tol

        self._vis = vis
        self.vis = vis.vis.isel(time=0)
        self.weight = vis.weight.isel(time=0)
        self.flags = vis.flags.isel(time=0)
        self.modelvis = modelvis.vis.isel(time=0)

        self.mask0 = self.antenna1 != self.antenna2
        self.wl_const = 2.0 * np.pi * const.c.value / vis.frequency.data

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

        self.cluster_indexes = cluster_indexes
        self.zernike_limit = zernike_limit

        self.solve_function = (
            self.solve_for_block_diagonal
            if block_diagonal
            else self.solve_for_non_block_diagonal
        )

    def solve(self):
        gaintable = create_bandpass_table(self._vis)
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
        modelvis = self.modelvis
        param = da.from_array(self.param)

        for it in range(self.niter):
            param_update = da.from_delayed(
                self.solve_function(modelvis, param, it),
                (n_cluster, n_param),
                np.float64,
            )

            param = param_update + param

            modelvis = da.from_delayed(
                self.apply_phase_distortions(modelvis, param_update),
                modelvis.shape,
                modelvis.dtype,
            )

        return param

    @dask.delayed
    def solve_for_block_diagonal(self, modelvis, param, it):
        [AA, Ab] = self.build_normal_equation(modelvis, param)

        n_cluster = np.amax(self.cluster_indexes) + 1
        [_, pidx0] = get_param_count(param)

        soln_vec = numpy.linalg.lstsq(AA, Ab, rcond=None)[0]

        param_update = np.zeros((n_cluster, param.shape[-1]), param.dtype)

        if self.change < self.tol:
            return param_update

        nu = 0.5
        for cid in range(n_cluster):
            param_update[cid] = (
                nu
                * soln_vec[
                    pidx0[cid] : pidx0[cid] + len(param[cid])
                ]  # noqa: E203
            )

        self.update_and_log_change(param_update, param + param_update, it)
        return param_update

    @dask.delayed
    def solve_for_non_block_diagonal(self, modelvis, param, it):
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

        self.update_and_log_change(param_update, param + param_update, it)
        return param_update

    def update_and_log_change(self, param_update, param, it):
        mask = np.abs(np.hstack(param).astype("float_")) > 0.0
        self.change = np.max(
            np.abs(np.hstack(param_update)[mask].astype("float_"))
            / np.abs(np.hstack(param)[mask].astype("float_"))
        )

        log.info(
            "Ionospheric Solver: Iteration %d, change: %f", it, self.change
        )

    def build_normal_equation(
        self,
        modelvis,
        param,
        cid=None,
    ):

        A = self.build_cluster_design_matrix(modelvis, param, cid)

        (n_param, _, n_freq, n_pol) = A.shape

        AA = np.zeros((n_param, n_param))
        Ab = np.zeros(n_param)

        wgt = self.weight[..., self.pols] * (1 - self.flags[..., self.pols])
        vis_diff = self.vis[..., self.pols] - modelvis[..., self.pols]

        for freq, pol in np.ndindex(n_freq, len(self.pols)):
            AA += np.real(
                np.einsum(
                    "pb,b,qb ->pq",
                    np.conj(A[..., freq, self.pols[pol]]),
                    wgt[..., freq, pol],
                    A[..., freq, pol],
                )
            )

            Ab += np.imag(
                np.einsum(
                    "pb,b,b->p",
                    np.conj(A[..., freq, self.pols[pol]]),
                    wgt[..., freq, pol],
                    vis_diff[..., freq, pol],
                )
            )

        return AA, Ab

    def build_cluster_design_matrix(
        self,
        modelvis,
        param,
        cid=None,
    ):
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
    def apply_phase_distortions(self, vis, param):
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
            vis[mask, :, 0] *= np.exp(
                # combine parmas for [n_baseline] then scale for [n_freq]
                np.einsum(
                    "b,f->bf",
                    (
                        # combine parmas for ant i in baselines
                        np.einsum(
                            "bp,p->b",
                            np.vstack(self.coeff[self.antenna1[mask]]).astype(
                                "float_"
                            ),
                            param[cid1],
                        )
                        # combine parmas for ant j in baselines
                        - np.einsum(
                            "bp,p->b",
                            np.vstack(self.coeff[self.antenna2[mask]]).astype(
                                "float_"
                            ),
                            param[cid2],
                        )
                    ),
                    1j * self.wl_const,
                )
            )

        return vis

    @dask.delayed
    def updated_gain_table(self, param, gain):
        [n_cluster, cid2stn, _] = set_cluster_maps(self.cluster_indexes)
        table_data = np.copy(gain.data)

        for cid in range(0, n_cluster):
            # combine parmas for [n_station] phase terms then scale for [n_freq]
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
