import dask.array as da
import numpy as np
import xarray as xr
from mock import ANY, MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.xarray_processors.rotation_measures import (  # noqa: E501
    RotationMeasureData,
    fit_curve,
    get_rm_spec,
    get_stn_masks,
    model_rotations,
    update_jones_with_masks,
)


def setup_test_data():
    nstations = 5
    nfreq = 10
    refant = 0

    gain_data = np.random.rand(
        1, nstations, nfreq, 2, 2
    ) + 1j * np.random.rand(1, nstations, nfreq, 2, 2)

    gain_data[:, :, :, 0, 0] += 0.5
    gain_data[:, :, :, 1, 1] += 0.5

    antenna_coords = [f"{i}" for i in range(nstations)]
    freq_coords = np.linspace(1e8, 2e8, nfreq)

    mock_gaintable = xr.Dataset(
        {
            "gain": (
                ("time", "antenna", "frequency", "receptor1", "receptor2"),
                gain_data,
            ),
            "weight": (
                ("time", "antenna", "frequency", "receptor1", "receptor2"),
                np.ones((1, nstations, nfreq, 2, 2)),
            ),
        },
        coords={
            "time": [0],
            "antenna": antenna_coords,
            "frequency": freq_coords,
        },
    )
    return nstations, nfreq, refant, mock_gaintable


def test_model_rotation_data_initialization():
    nstations, nfreq, refant, mock_gaintable = setup_test_data()

    rot_data = model_rotations(mock_gaintable, refant)

    assert len(rot_data.antenna) == nstations
    assert len(rot_data.frequency) == nfreq

    expected_lambda_sq = [
        8.98755179,
        7.27991695,
        6.01646029,
        5.05549788,
        4.30764316,
        3.71424334,
        3.23551864,
        2.84371756,
        2.5190024,
        2.24688795,
    ]

    np.testing.assert_allclose(rot_data.lambda_sq, expected_lambda_sq)

    assert rot_data.J.shape == (1, nstations, nfreq, 2, 2)
    assert rot_data.rm_est.shape == (
        1,
        nstations,
    )
    assert rot_data.rm_peak.shape == (
        1,
        nstations,
    )
    assert rot_data.const_rot.shape == (
        1,
        nstations,
    )
    assert rot_data.rm_spec is not None


def xtest_should_generate_rm_spec():
    phi_raw = np.zeros((2, 3))
    mask = np.array([[True, True, False], [False, True, True]])
    phasor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    expected = np.array(
        [[1.5, 4.5], [2.5, 5.5]], dtype=np.float64
    )  # pylint: disable=no-member

    out = get_rm_spec(  # pylint: disable=no-member
        phi_raw, mask, phasor
    ).compute()  # pylint: disable=no-member
    np.testing.assert_allclose(out, expected)


def test_should_calculate_phi_raw():
    nstations = 2
    nfreq = 2

    jones = np.ones((nstations, nfreq, 2, 2), dtype=complex)
    mask = np.array([[True, False], [True, True]])
    norms = np.full((nstations, nfreq, 2, 2), 2.0)

    jones_expected = [
        [
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            ],
            [
                [
                    1.0 + 0.0j,
                    1.0 + 0.0j,
                ],
                [1.0 + 0.0j, 1.0 + 0.0j],
            ],
        ],
        [
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            ],
            [
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
                [0.70710678 + 0.0j, 0.70710678 + 0.0j],
            ],
        ],
    ]

    out = update_jones_with_masks(  # pylint: disable=no-member
        jones, mask, norms
    )  # pylint: disable=no-member
    np.testing.assert_allclose(out, jones_expected)


def test_get_station_masks_when_refant_weights_are_all_zeros():
    ntime, nstn, nfreq, npol1, npol2 = (1, 3, 2, 2, 2)
    weight = np.zeros((ntime, nstn, nfreq, npol1, npol2))
    weight[0, :, :, 0, 0] = 1
    weight[0, :, :, 1, 1] = 1
    weight[0, 1, :, 0, 0] = 1
    weight[0, 1, :, 1, 1] = 1
    refant = 1

    out = get_stn_masks(weight[0, 0, ...], weight[0, refant, ...])
    np.testing.assert_array_equal(out, [True, True])


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".rotation_measures.curve_fit"
)
def test_should_fit_curve(curve_fit_mock):
    lambda_sq_mock = Mock(name="lambda sq")
    exp_stack = [-1, 0]
    rm_est_mock = [Mock(name="rm est 0"), Mock(name="rm est 1")]

    curve_fit_mock.side_effect = [
        (np.array([2.0, 1.0]), None),  # pylint: disable=no-member
        (np.array([3.0, 1.5]), None),  # pylint: disable=no-member
    ]

    out = fit_curve(  # pylint: disable=no-member
        lambda_sq_mock, [np.pi], rm_est_mock
    )

    expected = np.array([2.0, 1.0])

    curve_fit_mock.assert_called_once_with(ANY, lambda_sq_mock, ANY, p0=ANY)
    assert all(curve_fit_mock.call_args.args[2].astype(int) == exp_stack)
    assert curve_fit_mock.call_args.kwargs["p0"][0] == rm_est_mock
    assert curve_fit_mock.call_args.kwargs["p0"][1] == 0
    np.testing.assert_allclose(out, expected)
