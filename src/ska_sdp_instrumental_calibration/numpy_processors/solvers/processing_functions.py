from typing import Tuple

import numpy
import scipy
from ska_sdp_func_python.calibration.solvers import (
    solve_antenna_gains_itsubs_matrix,
    solve_antenna_gains_itsubs_nocrossdata,
    solve_antenna_gains_itsubs_scalar,
)


def find_best_refant_from_vis(
    flagged_vis: numpy.ndarray,
    flagged_weight: numpy.ndarray,
    ant1: numpy.ndarray,
    ant2: numpy.ndarray,
    nants: int,
):
    """
    Determine the best reference antenna based on visibility quality.

    This function ranks antennas by analyzing the Peak-to-Noise Ratio (PNR)
    of the delay transform (FFT of visibilities) for each antenna's
    baselines. For single-channel data, it ranks antennas based on the
    sum of their weights.

    Parameters
    ----------
    flagged_vis : numpy.ndarray
        Observed visibilities, typically with flags applied (i.e., bad data
        zeroed or masked). Shape: (ntime, nbl, nchan, npol).
    flagged_weight : numpy.ndarray
        Weights associated with the visibilities. Used primarily for the
        single-channel case. Shape: (ntime, nbl, nchan, npol).
    ant1 : numpy.ndarray
        Indices of antenna 1 for each baseline. Shape: (nbl,).
    ant2 : numpy.ndarray
        Indices of antenna 2 for each baseline. Shape: (nbl,).
    nants : int
        Total number of antennas in the array.

    Returns
    -------
    numpy.ndarray
        Array of antenna indices sorted in descending order of quality
        (best reference antenna first).

    Notes
    -----
    This method is adapted from `katsdpcal` [1]_.

    Algorithm details:

    * **Multi-channel:**

        1.  FFTs visibilities to the delay domain.
        2.  Centers the peak response.
        3.  Calculates PNR = (Peak - Mean) / Std, where Mean and Std are
            derived from the "noise" region (central channels of the FFT).
        4.  Antennas are ranked by the median PNR of all their baselines.

    * **Single-channel:**

        1.  Antennas are ranked by the sum of weights on their baselines.
        2.  A small epsilon is added to ensure stable sorting.

    References
    ----------
    .. [1] https://github.com/ska-sa/katsdpcal
    """
    visdata = flagged_vis
    _, _, nchan, _ = visdata.shape
    med_pnr_ants = numpy.zeros((nants))
    if nchan == 1:
        weightdata = flagged_weight
        for a in range(nants):
            mask = (ant1 == a) ^ (ant2 == a)
            weightdata_ant = weightdata[:, mask]
            mean_of_weight_ant = numpy.sum(weightdata_ant)
            med_pnr_ants[a] = mean_of_weight_ant
        med_pnr_ants += numpy.linspace(1e-8, 1e-9, nants)
    else:
        ft_vis = scipy.fftpack.fft(visdata, axis=2)
        max_value_arg = numpy.argmax(numpy.abs(ft_vis), axis=2)
        index = numpy.array(
            [numpy.roll(range(nchan), -n) for n in max_value_arg.ravel()]
        )
        index = index.reshape(list(max_value_arg.shape) + [nchan])
        index = numpy.transpose(index, (0, 1, 3, 2))
        ft_vis = numpy.take_along_axis(ft_vis, index, axis=2)

        peak = numpy.max(numpy.abs(ft_vis), axis=2)

        chan_slice = numpy.s_[
            nchan // 2 - nchan // 4 : nchan // 2 + nchan // 4 + 1  # noqa E203
        ]
        mean = numpy.mean(numpy.abs(ft_vis[:, :, chan_slice]), axis=2)
        std = numpy.std(numpy.abs(ft_vis[:, :, chan_slice]), axis=2) + 1e-9
        for a in range(nants):
            mask = (ant1 == a) ^ (ant2 == a)

            pnr = (peak[:, mask] - mean[:, mask]) / std[:, mask]
            med_pnr = numpy.median(pnr)
            med_pnr_ants[a] = med_pnr
    return numpy.argsort(med_pnr_ants)[::-1]


def gain_substitution(
    gain: numpy.ndarray,
    gain_weight: numpy.ndarray,
    gain_residual: numpy.ndarray,
    pointvis_vis: numpy.ndarray,
    pointvis_flags: numpy.ndarray,
    pointvis_weight: numpy.ndarray,
    ant1: numpy.ndarray,
    ant2: numpy.ndarray,
    crosspol: bool = False,
    niter: int = 30,
    phase_only: bool = True,
    tol: float = 1e-6,
    refant: int = 0,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Solve for antenna gains using the iterative substitution algorithm.

    This function acts as the driver for the gain substitution solver. It
    accumulates visibility data into antenna-based correlation matrices,
    determines the optimal reference antenna based on signal-to-noise
    ratio, and invokes the iterative solver backend.



    Parameters
    ----------
    gain : numpy.ndarray
        Initial gain estimates. Shape: (ntime, nants, nchan, nrec, nrec).
        Acts as the starting point for the iteration.
    gain_weight : numpy.ndarray
        Weights associated with the input gains. Shape matches `gain`.
    gain_residual : numpy.ndarray
        Buffer to store the residuals of the gain solution. Shape matches
        `gain`.
    pointvis_vis : numpy.ndarray
        Visibilities calibrated by the model (i.e., V_obs / V_model),
        effectively treating the sky as a point source of unit flux.
        Shape: (ntime, nbl, nchan, npol).
    pointvis_flags : numpy.ndarray
        Boolean flags for the point source visibilities (True indicates
        flagged/bad data). Shape matches `pointvis_vis`.
    pointvis_weight : numpy.ndarray
        Weights associated with the point source visibilities.
        Shape matches `pointvis_vis`.
    ant1 : numpy.ndarray
        Indices of antenna 1 for each baseline. Shape: (nbl,).
    ant2 : numpy.ndarray
        Indices of antenna 2 for each baseline. Shape: (nbl,).
    crosspol : bool, optional
        If True, solve for cross-polarization gain terms. Default is False.
    niter : int, optional
        Maximum number of iterations for the solver. Default is 30.
    phase_only : bool, optional
        If True, solve for phase terms only (amplitude fixed to 1.0).
        Default is True.
    tol : float, optional
        Tolerance for convergence. Iteration stops if the change in
        solution is below this threshold. Default is 1e-6.
    refant : int, optional
        Preferred reference antenna index. If this antenna is flagged or
        has low SNR, the solver will select a better fallback based on
        data quality. Default is 0.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:

        - Updated gain array.
        - Gain weights.
        - Gain residuals.
    """
    _gain = gain.copy()
    _gain_weight = gain_weight.copy()
    _gain_residual = gain_residual.copy()

    nants = _gain.shape[
        1
    ]  # TODO: should get antennas number from Configuration?
    nchan = _gain.shape[2]
    npol = pointvis_vis.shape[-1]

    axes = (0, 2) if nchan == 1 else 0

    pointvis_flagged_vis = _apply_flag(pointvis_vis, pointvis_flags)
    pointvis_flagged_weight = _apply_flag(pointvis_weight, pointvis_flags)

    refant_sort = find_best_refant_from_vis(
        pointvis_flagged_vis, pointvis_flagged_weight, ant1, ant2, nants
    )
    x_b = numpy.sum(
        (pointvis_vis * pointvis_weight) * (1 - pointvis_flags),
        axis=axes,
    )
    xwt_b = numpy.sum(
        pointvis_weight * (1 - pointvis_flags),
        axis=axes,
    )
    x = numpy.zeros([nants, nants, nchan, npol], dtype="complex")
    xwt = numpy.zeros([nants, nants, nchan, npol])

    for ibaseline, (a1, a2) in enumerate(zip(ant1, ant2)):
        x[a1, a2, ...] = numpy.conjugate(x_b[ibaseline, ...])
        xwt[a1, a2, ...] = xwt_b[ibaseline, ...]
        x[a2, a1, ...] = x_b[ibaseline, ...]
        xwt[a2, a1, ...] = xwt_b[ibaseline, ...]

    mask = numpy.abs(xwt) > 0.0
    if numpy.sum(mask) > 0:
        x_shape = x.shape
        x[mask] = x[mask] / xwt[mask]
        x[~mask] = 0.0
        xwt[mask] = xwt[mask] / numpy.max(xwt[mask])
        xwt[~mask] = 0.0
        x = x.reshape(x_shape)

        mask_solver = _get_mask_solver(crosspol, npol)

        (
            _gain[0, ...],
            _gain_weight[0, ...],
            _gain_residual[0, ...],
        ) = mask_solver(
            _gain[0, ...],
            _gain_weight[0, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )
    else:
        _gain[...] = 1.0 + 0.0j
        _gain_weight[...] = 0.0
        _gain_residual[...] = 0.0

    return (_gain, _gain_weight, _gain_residual)


def divide_visibility(
    vis_flagged: numpy.ndarray,
    vis_flagged_weight: numpy.ndarray,
    model_flagged_vis: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Divide visibility by model to form equivalent point source visibilities.

    This function computes the ratio of observed visibilities to model
    visibilities. This operation removes variations in time and frequency
    caused by the source structure, effectively normalizing the data to
    represent a point source of unit flux. This is a critical intermediate
    step in calibration, allowing for data averaging limited only by
    instrumental stability.

    The weights are adjusted to compensate for the division operation
    (propagation of variance), and protection against division by zero is
    included.

    Parameters
    ----------
    vis_flagged : numpy.ndarray
        Observed visibilities, typically with flags applied.
        Shape: (ntime, nbl, nchan, npol).
    vis_flagged_weight : numpy.ndarray
        Weights associated with the observed visibilities.
        Shape: matches `vis_flagged`.
    model_flagged_vis : numpy.ndarray
        Model visibilities corresponding to the observations.
        Shape: matches `vis_flagged`.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:

        - `x`: The divided visibilities (point source equivalent).
        - `xwt`: Adjusted weights corresponding to the divided visibilities.
    """
    x = numpy.zeros_like(vis_flagged)
    xwt = numpy.abs(model_flagged_vis) ** 2 * vis_flagged_weight
    mask = xwt > 0.0
    x[mask] = vis_flagged[mask] / model_flagged_vis[mask]

    return (x, xwt)


def create_point_vis(
    vis_vis: numpy.ndarray,
    vis_flags: numpy.ndarray,
    vis_weight: numpy.ndarray,
    model_vis: numpy.ndarray,
    model_flags: numpy.ndarray,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Create equivalent point source visibilities from observed and model data.

    This function prepares visibility data for calibration by removing source
    structure effects. If a model is provided, the observed visibilities are
    divided by the model visibilities (using `divide_visibility`). Flags are
    applied to both observed and model data before this operation.

    If `model_vis` is None, the observed visibilities are returned as-is,
    assuming they effectively represent a point source or no model correction
    is required.

    Parameters
    ----------
    vis_vis : numpy.ndarray
        Observed complex visibilities. Shape: (ntime, nbl, nchan, npol).
    vis_flags : numpy.ndarray
        Boolean flags for observed data (True indicates flagged/bad data).
        Shape matches `vis_vis`.
    vis_weight : numpy.ndarray
        Weights associated with the observed visibilities.
        Shape matches `vis_vis`.
    model_vis : numpy.ndarray or None
        Model complex visibilities representing the source structure.
        If None, the function bypasses the division step.
    model_flags : numpy.ndarray or None
        Boolean flags for the model data. Must be provided if `model_vis`
        is provided.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:

        - `point_vis`: The resulting point source equivalent visibilities.
        - `point_weight`: The associated wghts (adjusted if division occurred)
    """
    vis_flagged = _apply_flag(vis_vis, vis_flags)
    vis_flagged_weight = _apply_flag(vis_weight, vis_flags)

    return (
        divide_visibility(
            vis_flagged,
            vis_flagged_weight,
            _apply_flag(model_vis, model_flags),
        )
        if model_vis is not None
        else (vis_vis, vis_weight)
    )


def _get_mask_solver(crosspol, npol):
    if npol == 2 or (npol == 4 and not crosspol):
        return solve_antenna_gains_itsubs_nocrossdata

    if npol == 4:
        return solve_antenna_gains_itsubs_matrix

    return solve_antenna_gains_itsubs_scalar


def _apply_flag(x: numpy.ndarray, flags: numpy.ndarray):
    return x * (1 - flags)
