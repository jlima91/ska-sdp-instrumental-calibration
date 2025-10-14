# flake8: noqa
# pylint: skip-file
import logging

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
        self, vis, modelvis, cluster_indexes=None, block_diagonal=False
    ):

        if np.all(modelvis.vis == 0.0):
            raise ValueError("solve_ionosphere: Model visibilities are zero")

        self.xyz = vis.configuration.xyz.data
        self.antenna1 = vis.antenna1.data
        self.antenna2 = vis.antenna2.data

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

        self.block_diagonal_solve_function = (
            self.__solve_for_block_diagonal
            if block_diagonal
            else self.__solve_for_non_block_diagonal
        )

        self.vis = vis
        self.modelvis = modelvis

    def solve(self, zernike_limit=None, niter=15, tol=1e-6, chunks=None):
        chunks = {} if chunks is None else chunks

        restored_vis = restore_baselines_dim(self.vis)
        restored_modelvis = restore_baselines_dim(self.modelvis)

        gaintable = (
            create_bandpass_table(self.vis)
            .pipe(with_chunks, chunks)
            .rename({"time": "soln_time"})
        )

        if self.cluster_indexes is None:
            self.cluster_indexes = np.zeros(len(gaintable.antenna), "int")

        if len(gaintable.antenna) != len(self.cluster_indexes):
            raise ValueError(
                f"cluster_indexes has wrong size {len(self.cluster_indexes)}"
            )

        return gaintable.map_blocks(
            self.compute_gains,
            args=[restored_vis, restored_modelvis, zernike_limit, niter, tol],
            template=gaintable,
        ).rename({"soln_time": "time"})

    def compute_gains(
        self, gain_chunk, vis, modelvis, zernike_limit=None, niter=15, tol=1e-6
    ):
        param, coeff = self.__compute_param_coeff(
            gain_chunk, vis, modelvis, zernike_limit, niter, tol
        )

        return self.update_gain_table(gain_chunk, param, coeff)

    def update_gain_table(
        self,
        gain_chunk,
        param,
        coeff,
    ):
        """
        Add new solutions to gaintable.

        Expand solutions for all stations and frequency channels and insert in the
        gain table

        :param gain_table: GainTable to be updated
        :param param: [n_cluster] list of solution vectors, one for each cluster
        :param coeff: [n_station] list of basis-func value vectors, one per station
            Stored as a numpy dtype=object array of variable-length coeff vectors
        :param cluster_id: [n_antenna] array of antenna cluster indices

        """
        # Get common mapping vectors between stations and clusters
        [n_cluster, cid2stn, _] = set_cluster_maps(self.cluster_indexes)
        wl_const = 2.0 * np.pi * const.c.value / gain_chunk.frequency.data

        table_data = np.copy(gain_chunk.gain.data)

        for cid in range(0, n_cluster):
            # combine parmas for [n_station] phase terms then scale for [n_freq]
            table_data[0, cid2stn[cid], :, 0, 0] = np.exp(
                np.einsum(
                    "s,f->sf",
                    np.einsum(
                        "sp,p->s",
                        np.vstack(coeff[cid2stn[cid]]).astype("float_"),
                        param[cid],
                    ),
                    1j * wl_const,
                )
            )

        new_gain = gain_chunk.gain.copy()
        new_gain.data = table_data

        return gain_chunk.assign(
            {
                "gain": new_gain,
            }
        )

    def apply_phase_distortions(self, gain_chunk, vis, param, coeff):
        """
        Update visibility model with new fit solutions.

        :param vis: Visibility containing the data_models to be distorted
        :param param: [n_cluster] list of solution vectors, one for each cluster
        :param coeff: [n_station] list of basis-func value vectors, one per station
            Stored as a numpy dtype=object array of variable-length coeff vectors
        :param cluster_id: [n_antenna] array of antenna cluster indices

        """
        # Get common mapping vectors between stations and clusters
        [n_cluster, _, stn2cid] = set_cluster_maps(self.cluster_indexes)

        # set up a few references and constants

        # exclude auto-correlations from the mask
        mask0 = self.antenna1 != self.antenna2
        wl_const = 2.0 * np.pi * const.c.value / gain_chunk.frequency.data

        # Use einsum calls to average over parameters for all combinations of
        # baseline and frequency
        # [n_freq] scaling constants
        # Loop over pairs of clusters and update the associated baselines
        for cid1 in range(0, n_cluster):
            for cid2 in range(0, n_cluster):
                # A mask for all baselines in this cluster pair
                mask = (
                    mask0
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
                                np.vstack(coeff[self.antenna1[mask]]).astype(
                                    "float_"
                                ),
                                param[cid1],
                            )
                            # combine parmas for ant j in baselines
                            - np.einsum(
                                "bp,p->b",
                                np.vstack(coeff[self.antenna2[mask]]).astype(
                                    "float_"
                                ),
                                param[cid2],
                            )
                        ),
                        # phase scaling with frequency
                        1j * wl_const,
                    )
                )

        return vis

    def build_normal_equation(
        self,
        gain_chunk,
        vis,
        modelvis,
        coeff,
        param,
        cid=None,
    ):
        vis_data = vis.vis.data[0, ...]
        weight = vis.weight[0, ...]
        flags = vis.flags[0, ...]

        A = self.__build_cluster_design_matrix(
            gain_chunk, modelvis, coeff, param, cid
        )
        (n_param, _, n_freq, n_pol) = A.shape
        AA = np.zeros((n_param, n_param))
        Ab = np.zeros(n_param)
        wgt = weight[..., self.pols] * (1 - flags[..., self.pols])
        vis_diff = vis_data[..., self.pols] - modelvis[..., self.pols]
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

    def cluster_design_matrix(
        self,
        gain_chunk,
        modelvis,
        mask0,
        coeff,
        stn2cid,
        n_param,
        cid,
    ):
        """
        Generate elements of the design matrix for the current cluster.

        Dereference outside of loops and the function call to avoid overheads.

        :param modelvis: [n_time,n_baseline,n_pol] predicted model vis for chan
        :param mask0: [n_baseline] mask of wanted data samples
        :param antenna1: [n_baseline] station index of first antenna in each baseline
        :param antenna2: [n_baseline] station index of second antenna in each baseline
        :param coeff: [n_station] list of basis-func value vectors, one per station
        :param stn2cid: [n_station] mapping from station index to cluster index
        :param wl_const: 2*pi*lambda for the current frequency channel
        :param cid: index of current cluster
        :param n_param: number of parameters in Normal equation

        """
        wl_const = 2.0 * np.pi * const.c.value / gain_chunk.frequency.data
        n_baselines = len(mask0)
        A = np.zeros((n_param, *modelvis.shape), "complex_")
        wl_const = wl_const.reshape(1, *wl_const.shape, 1)

        blidx_all = np.arange(n_baselines)
        blidx = blidx_all[mask0 * (stn2cid[self.antenna1] == cid)]

        # Get all masked baselines with antenna1 in this cluster
        if len(blidx) > 0:
            # [nvis] A0 terms x [nvis,nparam] coeffs (1st antenna)
            # all masked antennas have the same number of coeffs so can vstack
            A[:, blidx, :] += np.einsum(
                "bfq,bp->pbfq",
                modelvis[blidx, :, :] * wl_const,
                np.vstack(coeff[self.antenna1[blidx]]).astype(np.float32),
            )

        # Get all masked baselines with antenna2 in this cluster
        blidx = blidx_all[mask0 * (stn2cid[self.antenna2] == cid)]
        if len(blidx) > 0:
            # [nvis] A0 terms x [nvis,nparam] coeffs (2nd antenna)
            # all masked antennas have the same number of coeffs so can vstack
            A[:, blidx, :] -= np.einsum(
                "bfq,bp->pbfq",
                modelvis[blidx, :, :] * wl_const,
                np.vstack(coeff[self.antenna2[blidx]]).astype(np.float32),
            )

        return A

    def __compute_param_coeff(
        self,
        gain_chunk,
        vis,
        modelvis,
        zernike_limit,
        niter,
        tol,
    ):

        n_cluster = np.amax(self.cluster_indexes) + 1
        modelvis = np.copy(modelvis.vis.data[0, ...])

        [param, coeff] = set_coeffs_and_params(
            self.xyz, self.cluster_indexes, zernike_limit
        )

        n_param = get_param_count(param)[0]
        if n_cluster == 1:
            log.info(
                "Setting up iono solver for %d stations in a single cluster",
                len(gain_chunk.antenna),
            )
            log.info("There are %d total parameters in the cluster", n_param)
        else:
            log.info(
                "Setting up iono solver for %d stations in %d clusters",
                len(gain_chunk.antenna),
                n_cluster,
            )
            log.info(
                "There are %d total parameters: %d in c[0] + %d x c[1:%d]",
                n_param,
                len(param[0]),
                len(param[1]),
                len(param) - 1,
            )

        for it in range(niter):
            param_update = self.block_diagonal_solve_function(
                gain_chunk, vis, modelvis, param, coeff, it
            )

            modelvis = self.apply_phase_distortions(
                gain_chunk, modelvis, param_update, coeff
            )

            mask = np.abs(np.hstack(param).astype("float_")) > 0.0
            change = np.max(
                np.abs(np.hstack(param_update)[mask].astype("float_"))
                / np.abs(np.hstack(param)[mask].astype("float_"))
            )

            log.info(
                "Ionospheric Solver: Iteration %d, change: %f", it, change
            )

            if change < tol:
                break

        return (param, coeff)

    def __build_cluster_design_matrix(
        self,
        gain_chunk,
        modelvis,
        coeff,
        param,
        cid=None,
    ):
        # If no cluster index is given, build matrix for all clusters
        generate_full_equation = cid is None

        # Get common mapping vectors between stations and clusters
        [n_cluster, _, stn2cid] = set_cluster_maps(self.cluster_indexes)

        # Exclude auto-correlations from the mask
        mask = self.antenna1 != self.antenna2

        if not generate_full_equation:

            n_param = len(param[cid])

            return self.cluster_design_matrix(
                gain_chunk,
                modelvis,
                mask,
                coeff,
                stn2cid,
                n_param,
                cid,
            )

        [n_param, pidx0] = get_param_count(param)

        A = np.zeros((n_param, *modelvis.shape), "complex_")

        for _cid in range(0, n_cluster):
            pid = np.arange(pidx0[_cid], pidx0[_cid] + len(param[_cid]))

            A[pid, :] += self.cluster_design_matrix(
                gain_chunk,
                modelvis,
                mask,
                coeff,
                stn2cid,
                len(param[_cid]),
                _cid,
            )

        return A

    def __solve_for_block_diagonal(
        self, gain_chunk, vis, modelvis, param, coeff, it
    ):  # `param` is updated by ref
        [AA, Ab] = self.build_normal_equation(
            gain_chunk, vis, modelvis, coeff, param
        )

        # Solve the normal equations and update parameters
        return solve_normal_equation(AA, Ab, param, it)

    def __solve_for_non_block_diagonal(
        self, gain_chunk, vis, modelvis, param, coeff, it
    ):  # `param` is updated by ref

        n_cluster = np.amax(self.cluster_indexes) + 1
        param_update = []
        for cid in range(n_cluster):
            [AA, Ab] = self.build_normal_equation(
                gain_chunk, vis, modelvis, coeff, param, cid
            )

            # Solve the current incremental normal equations
            soln_vec = np.linalg.lstsq(AA, Ab, rcond=None)[0]

            # Update factor
            nu = 0.5
            # nu = 1.0 - 0.5 * (it % 2)
            param_update.append(nu * soln_vec)
            param[cid] += param_update[cid]

        return param_update
