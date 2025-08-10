import dask.array as da
import numpy as np
import xarray as xr
from mock import ANY, MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.processing_tasks.rotation_measures import (  # noqa: E501
    ModelRotationData,
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
                ("time", "antenna", "frequency", "pol1", "pol2"),
                gain_data,
            ),
            "weight": (
                ("time", "antenna", "frequency"),
                np.ones((1, nstations, nfreq)),
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

    rot_data = ModelRotationData(mock_gaintable, refant)

    assert rot_data.nstations == nstations
    assert rot_data.nfreq == nfreq
    assert rot_data.refant == refant

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

    assert rot_data.rm_res > 0
    assert rot_data.rm_max > 0
    assert len(rot_data.rm_vals) > 0
    assert rot_data.phasor.shape == (len(rot_data.rm_vals), nfreq)

    assert rot_data.J.shape == (nstations, nfreq, 2, 2)
    assert rot_data.rm_est.shape == (nstations,)
    assert rot_data.rm_peak.shape == (nstations,)
    assert rot_data.const_rot.shape == (nstations,)
    assert rot_data.rm_spec is None


def test_model_rotation_data_value_error():
    nstations, nfreq, refant, _ = setup_test_data()

    incorrect_gain_data = np.random.rand(1, nstations, nfreq, 1, 1)
    incorrect_gaintable = xr.Dataset(
        {
            "gain": (
                ("time", "antenna", "frequency", "pol1", "pol2"),
                incorrect_gain_data,
            )
        },
        coords={
            "time": [0],
            "antenna": [f"ant{i}" for i in range(nstations)],
            "frequency": np.linspace(1e8, 2e8, nfreq),
        },
    )
    exception_caught = False
    try:
        ModelRotationData(incorrect_gaintable, refant)
    except ValueError as e:
        exception_caught = True
        assert str(e) == "gaintable must contain Jones matrices"
    assert exception_caught, "ValueError was not raised when expected"


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.ModelRotationData"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.calculate_phi_raw"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.get_stn_masks"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.update_jones_with_masks"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.get_rm_spec"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.fit_curve"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.from_delayed"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.linalg.norm"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.abs"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.max"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.argmax"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.where"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.cos"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.sin"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks."
    "rotation_measures.dask.array.hstack"
)
def test_model_rotations_function(
    mock_dask_hstack,
    mock_dask_sin,
    mock_dask_cos,
    mock_dask_where,
    mock_dask_argmax,
    mock_dask_max,
    mock_dask_abs,
    mock_norm,
    mock_from_delayed,
    mock_fit_curve,
    mock_get_rm_spec,
    mock_update_jones_with_masks,
    mock_get_stn_masks,
    mock_calculate_phi_raw,
    MockModelRotationData,
):

    mock_gaintable = MagicMock(spec=xr.Dataset)
    mock_gaintable.weight = MagicMock(name="gaintable_weight")

    mock_rotations_instance = MagicMock(spec=ModelRotationData)
    mock_rotations_instance.nstations = 3
    mock_rotations_instance.nfreq = 5

    mock_rotations_instance.rm_vals = MagicMock(
        spec=da.Array, name="mock_rm_vals"
    )
    mock_rotations_instance.phasor = MagicMock(
        spec=da.Array, name="mock_phasor"
    )
    mock_rotations_instance.lambda_sq = MagicMock(
        spec=da.Array, name="mock_lambda_sq"
    )
    mock_rotations_instance.J = MagicMock(
        spec=da.Array, name="mock_J"
    )  # noqa: E501
    MockModelRotationData.return_value = mock_rotations_instance
    mock_norms_dask_array = MagicMock(
        spec=da.Array, name="mock_norms_dask_array"
    )
    mock_norms_dask_array.__getitem__.return_value = MagicMock(  # noqa: E501
        spec=da.Array, name="mock_norms_slice"
    )
    mock_norms_dask_array.__getitem__.return_value.__gt__.return_value = (
        MagicMock(spec=da.Array, name="mock_norms_gt_result")  # noqa: E501
    )
    mock_norm.return_value = mock_norms_dask_array
    mock_mask_dask_array = MagicMock(
        spec=da.Array, name="mock_mask_dask_array"
    )
    mock_rm_spec_dask_array = MagicMock(
        spec=da.Array, name="mock_rm_spec_dask_array"
    )
    mock_fit_rm_dask_array = MagicMock(
        spec=da.Array, name="mock_fit_rm_dask_array"
    )

    mock_phi_raw = MagicMock(spec=da.Array, name="mock_phi_raw")

    mock_from_delayed.side_effect = [
        mock_mask_dask_array,
        mock_rotations_instance.J,
        mock_phi_raw,
        mock_rm_spec_dask_array,
        mock_fit_rm_dask_array,
    ]

    mock_mask_dask_array.__and__.return_value = MagicMock(
        spec=da.Array, name="final_mask_dask_array"
    )

    mock_fit_rm_dask_array.__getitem__.side_effect = lambda idx: MagicMock(
        spec=da.Array
    )  # noqa: E501

    mock_get_stn_masks.return_value = np.ones((3, 5), dtype=bool)
    mock_update_jones_with_masks.return_value = mock_rotations_instance.J
    mock_get_rm_spec.return_value = np.random.random((3, 199))
    mock_fit_curve.return_value = np.array([0.123, 0.456])

    mock_dask_abs.return_value = MagicMock(spec=da.Array)
    mock_dask_max_result = MagicMock(
        spec=da.Array, name="mock_dask_max_result"
    )

    mock_dask_max_result.__gt__.return_value = MagicMock(
        spec=da.Array, name="mock_dask_max_gt_result"
    )
    mock_dask_max.return_value = mock_dask_max_result
    mock_dask_argmax.return_value = MagicMock(spec=da.Array)

    mock_rotations_instance.rm_vals.__getitem__.return_value = MagicMock(
        spec=da.Array, name="mock_rm_vals_indexed_result"
    )
    mock_dask_where.return_value = MagicMock(spec=da.Array)
    mock_dask_cos.return_value = MagicMock(spec=da.Array)
    mock_dask_sin.return_value = MagicMock(spec=da.Array)
    mock_dask_hstack.return_value = MagicMock(spec=da.Array)

    model_rotations(
        mock_gaintable, peak_threshold=0.5, refine_fit=True, refant=1
    )

    MockModelRotationData.assert_called_once_with(mock_gaintable, 1, 5)
    mock_norm.assert_called_once_with(
        mock_rotations_instance.J, axis=(2, 3), keepdims=True
    )
    mock_norms_dask_array.__getitem__.assert_called_once_with(
        (slice(None), slice(None), 0, 0)
    )
    mock_norms_dask_array.__getitem__.return_value.__gt__.assert_called_once_with(  # noqa: E501
        0
    )
    mock_get_stn_masks.assert_called_once_with(mock_gaintable.weight, 1)

    mock_mask_dask_array.__and__.assert_called_once_with(
        mock_norms_dask_array.__getitem__.return_value.__gt__.return_value
    )
    mock_update_jones_with_masks.assert_called_once_with(
        mock_rotations_instance.J,
        mock_mask_dask_array.__and__.return_value,
        mock_norm.return_value,
        mock_rotations_instance.nstations,
    )

    mock_dask_abs.assert_called_with(
        mock_rm_spec_dask_array
    )  # pylint: disable=no-member
    mock_dask_max.assert_called_once_with(mock_dask_abs.return_value, axis=1)
    mock_dask_argmax.assert_called_once_with(
        mock_dask_abs.return_value, axis=1  # pylint: disable=no-member
    )
    mock_dask_where.assert_called_once_with(
        mock_dask_max_result.__gt__.return_value,  # pylint: disable=no-member
        mock_rotations_instance.rm_vals[mock_dask_argmax.return_value],
        0,
    )
    mock_rotations_instance.rm_peak = mock_dask_where.return_value
    mock_fit_curve.assert_called_once_with(
        mock_rotations_instance.lambda_sq,
        mock_dask_hstack.return_value,
        mock_rotations_instance.rm_peak,
        mock_rotations_instance.nstations,
    )


def test_should_generate_rm_spec():
    phasor = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    phi_raw = np.zeros((2, 3))

    mask = np.array([[True, True, False], [False, True, True]])

    expected = np.array(
        [[1.5, 4.5], [2.5, 5.5]], dtype=float
    )  # pylint: disable=no-member

    out = get_rm_spec(  # pylint: disable=no-member
        phi_raw, phasor, mask, nstations=2
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
        jones, mask, norms, nstations
    ).compute()  # pylint: disable=no-member
    np.testing.assert_allclose(out, jones_expected)


def test_get_station_masks_when_refant_weights_are_all_zeros():
    ntime, nstn, nfreq, npol1, npol2 = (1, 3, 2, 2, 2)
    weight = np.zeros((ntime, nstn, nfreq, npol1, npol2))
    weight[0, :, :, 0, 0] = 1
    weight[0, :, :, 1, 1] = 1
    weight[0, 1, :, 0, 0] = 1
    weight[0, 1, :, 1, 1] = 1
    refant = 1

    expected = (
        (weight[0, :, :, 0, 0] > 0)
        & (weight[0, :, :, 1, 1] > 0)
        & (weight[0, refant, :, 0, 0] > 0)
        & (weight[0, refant, :, 1, 1] > 0)
    )

    out = get_stn_masks(weight, refant).compute()
    np.testing.assert_array_equal(out, expected)


def test_get_station_masks_when_refant_weights_are_non_zero():
    ntime, nstn, nfreq, npol1, npol2 = (1, 2, 2, 2, 2)
    weight = np.ones((ntime, nstn, nfreq, npol1, npol2))
    weight[0, 0, :, 0, 1] = 2
    refant = 0

    expected = np.all(weight[0, :] > 0, axis=(2, 3)) & np.all(
        weight[0, refant] > 0, axis=(1, 2)
    )

    out = get_stn_masks(weight, refant).compute()
    np.testing.assert_array_equal(out, expected)


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks"
    ".rotation_measures.curve_fit"
)
def test_should_fit_curve(curve_fit_mock):
    nstations = 2
    lambda_sq_mock = Mock(name="lambda sq")
    exp_stack_mock = [Mock(name="exp stack 0"), Mock(name="exp stack 1")]
    rm_est_mock = [Mock(name="rm est 0"), Mock(name="rm est 1")]

    curve_fit_mock.side_effect = [
        (np.array([2.0, 1.0]), None),  # pylint: disable=no-member
        (np.array([3.0, 1.5]), None),  # pylint: disable=no-member
    ]

    out = fit_curve(  # pylint: disable=no-member
        lambda_sq_mock, exp_stack_mock, rm_est_mock, nstations
    ).compute()

    expected = np.array([[2.0, 3.0], [1.0, 1.5]])

    curve_fit_mock.assert_has_calls(
        [
            call(
                ANY, lambda_sq_mock, exp_stack_mock[0], p0=[rm_est_mock[0], 0]
            ),
            call(
                ANY, lambda_sq_mock, exp_stack_mock[1], p0=[rm_est_mock[1], 0]
            ),
        ]
    )

    assert curve_fit_mock.call_count == nstations
    np.testing.assert_allclose(out, expected)
