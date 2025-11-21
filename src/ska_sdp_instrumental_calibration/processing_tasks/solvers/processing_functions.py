from typing import Tuple

import numpy
import scipy
from ska_sdp_func_python.calibration.solvers import (
    _solve_antenna_gains_itsubs_matrix,
    _solve_antenna_gains_itsubs_nocrossdata,
    _solve_antenna_gains_itsubs_scalar,
)


def find_best_refant_from_vis(
    flagged_vis: numpy.ndarray,
    flagged_weight: numpy.ndarray,
    ant1: numpy.ndarray,
    ant2: numpy.ndarray,
    nants: int,
):
    """
    This method comes from katsdpcal.
    (https://github.com/ska-sa/katsdpcal/blob/
    200c2f6e60b2540f0a89e7b655b26a2b04a8f360/katsdpcal/calprocs.py#L332)
    Determine antenna whose FFT has the maximum peak to noise ratio (PNR) by
    taking the median PNR of the FFT over all baselines to each antenna.

    When the input vis has only one channel, this uses all the vis of the
    same antenna for the operations peak, mean and std.

    :param vis: Visibilities
    :return: Array of indices of antennas in decreasing order
            of median of PNR over all baselines

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


def _solve_with_mask(
    crosspol,
    gaintable_gain: numpy.ndarray,
    gaintable_weight: numpy.ndarray,
    gaintable_residual: numpy.ndarray,
    mask,
    niter,
    phase_only,
    row,
    tol,
    npol,
    x,
    xwt,
    refant,
    refant_sort,
):
    """
    Method extracted from solve_gaintable to decrease
    complexity. Calculations when `numpy.sum(mask) > 0`
    """
    x_shape = x.shape
    x[mask] = x[mask] / xwt[mask]
    x[~mask] = 0.0
    xwt[mask] = xwt[mask] / numpy.max(xwt[mask])
    xwt[~mask] = 0.0
    x = x.reshape(x_shape)
    if npol == 2 or (npol == 4 and not crosspol):
        (
            gaintable_gain[row, ...],
            gaintable_weight[row, ...],
            gaintable_residual[row, ...],
        ) = _solve_antenna_gains_itsubs_nocrossdata(
            gaintable_gain[row, ...],
            gaintable_weight[row, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )
    elif npol == 4 and crosspol:
        (
            gaintable_gain[row, ...],
            gaintable_weight[row, ...],
            gaintable_residual[row, ...],
        ) = _solve_antenna_gains_itsubs_matrix(
            gaintable_gain[row, ...],
            gaintable_weight[row, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )

    else:
        (
            gaintable_gain[row, ...],
            gaintable_weight[row, ...],
            gaintable_residual[row, ...],
        ) = _solve_antenna_gains_itsubs_scalar(
            gaintable_gain[row, ...],
            gaintable_weight[row, ...],
            x,
            xwt,
            phase_only=phase_only,
            niter=niter,
            tol=tol,
            refant=refant,
            refant_sort=refant_sort,
        )


def _apply_flag(x: numpy.ndarray, flags: numpy.ndarray):
    return x * (1 - flags)


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
        _solve_with_mask(
            crosspol,
            _gain,
            _gain_weight,
            _gain_residual,
            mask,
            niter,
            phase_only,
            0,
            tol,
            npol,
            x,
            xwt,
            refant,
            refant_sort,
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
):
    """
    Divide visibility by model forming
    visibility for equivalent point source.

    This is a useful intermediate product for calibration.
    Variation of the visibility in time and frequency due
    to the model structure is removed and the data can be
    averaged to a limit determined by the instrumental stability.
    The weight is adjusted to compensate for the division.

    Zero divisions are avoided and the corresponding weight set to zero.

    :param vis: Visibility to be divided
    :param modelvis: Visibility to divide with
    :return: Divided Visibility
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
):
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
