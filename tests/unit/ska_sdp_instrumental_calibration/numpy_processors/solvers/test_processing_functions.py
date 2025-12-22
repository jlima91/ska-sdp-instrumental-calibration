import numpy as np

from ska_sdp_instrumental_calibration.numpy_processors.solvers import (
    processing_functions,
)


def test_should_find_best_refant_from_vis():
    vis = np.arange(48).reshape(2, 2, 3, 4)
    weight = np.ones_like(vis)

    ant1 = np.arange(2)
    ant2 = np.arange(2)
    nants = 2

    actual = processing_functions.find_best_refant_from_vis(
        vis, weight, ant1, ant2, nants
    )

    assert all(actual == [1, 0])


def test_should_find_best_refant_from_vis_for_single_channel():
    vis = np.arange(16).reshape(2, 2, 1, 4)
    weight = np.ones_like(vis)

    ant1 = np.arange(2)
    ant2 = np.arange(2)
    nants = 2

    actual = processing_functions.find_best_refant_from_vis(
        vis, weight, ant1, ant2, nants
    )

    assert all(actual == [0, 1])


def test_should_perform_gain_substitution():
    gain = np.arange(24).reshape(1, 2, 3, 2, 2)
    gain_weight = np.ones_like(gain)
    gain_residual = np.zeros_like(gain)
    point_vis = np.ones((2, 2, 3, 4))
    poin_vis_flags = np.zeros_like(point_vis)
    point_vis_weight = np.ones_like(point_vis)

    ant1 = np.arange(2)
    ant2 = np.arange(2)
    niter = 2

    ac_gain, ac_weight, ac_residual = processing_functions.gain_substitution(
        gain,
        gain_weight,
        gain_residual,
        point_vis,
        poin_vis_flags,
        point_vis_weight,
        ant1,
        ant2,
        niter=niter,
    )

    ex_gain = np.array(
        [
            [
                [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
            ]
        ]
    )

    ex_weight = np.zeros_like(gain)
    ex_residual = np.zeros_like(gain)

    np.testing.assert_allclose(ac_gain, ex_gain)
    np.testing.assert_allclose(ac_weight, ex_weight)
    np.testing.assert_allclose(ac_residual, ex_residual)


def test_should_solve_with_mask_for_2_pol():
    assert (
        processing_functions._get_mask_solver(
            True,
            2,
        )
        == processing_functions._solve_antenna_gains_itsubs_nocrossdata
    )


def test_should_solve_with_mask_for_4_pol_and_no_crosspol():
    assert (
        processing_functions._get_mask_solver(
            False,
            4,
        )
        == processing_functions._solve_antenna_gains_itsubs_nocrossdata
    )


def test_should_solve_with_mask_for_4_pol_and_crosspol():
    assert (
        processing_functions._get_mask_solver(
            True,
            4,
        )
        == processing_functions._solve_antenna_gains_itsubs_matrix
    )


def test_should_solve_with_itsub_scalar_mask():
    assert (
        processing_functions._get_mask_solver(
            False,
            3,
        )
        == processing_functions._solve_antenna_gains_itsubs_scalar
    )


def test_should_perform_gain_substitution_for_no_mask():
    gain = np.arange(24).reshape(1, 2, 3, 2, 2)
    gain_weight = np.ones_like(gain)
    gain_residual = np.zeros_like(gain)
    point_vis = np.zeros((2, 2, 3, 4))
    poin_vis_flags = np.zeros_like(point_vis)
    point_vis_weight = np.zeros_like(point_vis)
    gain = gain + 0.0j

    ant1 = np.arange(2)
    ant2 = np.arange(2)
    niter = 2

    ac_gain, ac_weight, ac_residual = processing_functions.gain_substitution(
        gain,
        gain_weight,
        gain_residual,
        point_vis,
        poin_vis_flags,
        point_vis_weight,
        ant1,
        ant2,
        niter=niter,
        crosspol=True,
    )

    ex_gain = np.ones_like(gain) + 0.0j

    ex_weight = np.zeros_like(gain_weight)
    ex_residual = np.zeros_like(gain_residual)

    np.testing.assert_allclose(ac_gain, ex_gain)
    np.testing.assert_allclose(ac_weight, ex_weight)
    np.testing.assert_allclose(ac_residual, ex_residual)
