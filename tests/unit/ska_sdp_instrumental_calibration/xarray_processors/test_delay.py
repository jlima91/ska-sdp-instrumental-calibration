# pylint: disable = too-many-function-args
import numpy as np
import xarray as xr
from mock import Mock, patch

from ska_sdp_instrumental_calibration.xarray_processors.delay import (
    apply_delay,
    calculate_delay,
    calculate_gain_rot,
    calibrate_polarization,
    coarse_delay,
    stack_jones_coordinate,
    unstack_jones_coordinate,
)


def test_should_calculate_coarse_delay():
    oversample = 8
    nstations = 20
    nchan = 4
    frequency = np.linspace(100e6, 150e6, nchan)
    gains_per_station = np.ones(nchan) + 1j * np.ones(nchan)
    gains = xr.DataArray(
        np.stack([gains_per_station] * nstations),
        dims=["antenna", "frequency"],
        coords={"antenna": np.arange(nstations), "frequency": frequency},
    )

    expected = np.zeros(nstations)
    actual = coarse_delay(gains, oversample)

    assert np.allclose(expected, actual, atol=1e-11)


def test_calculate_gain_rotation():
    frequency = np.linspace(100e6, 200e6, 4).reshape(1, -1)
    f = np.linspace(4, 10, 4) + 1j * np.linspace(3, 9, 4)
    gains = np.stack([f] * 2)
    delay = np.ones(2)
    offset = np.ones(2)

    expected = [
        [
            3.99999993 + 3.00000009j,
            1.33012709 - 7.69615241j,
            -10.06217822 + 3.42820208j,
            10.00000017 + 8.99999981j,
        ],
        [
            3.99999993 + 3.00000009j,
            1.33012709 - 7.69615241j,
            -10.06217822 + 3.42820208j,
            10.00000017 + 8.99999981j,
        ],
    ]
    actual = calculate_gain_rot(gains, delay, offset, frequency)

    assert np.allclose(actual, expected)


def test_calculate_apply_delay():
    coords = {
        "antenna": ["antenna1", "antenna2"],
        "frequency": np.linspace(1.0010e8, 1.0019e8, 4, dtype=np.float64),
    }

    # gain_shape = [ntimes, nants, nfrequency, nrec, nrec]
    gains = xr.DataArray(
        np.array(
            [
                np.cos(np.pi / 3) + 1j * np.sin(np.pi / 3),
                np.cos(2 * np.pi / 3) + 1j * np.sin(2 * np.pi / 3),
            ]
            * 16,
            dtype=np.complex128,
        ).reshape(1, 2, 4, 2, 2),
        dims=["time", "antenna", "frequency", "rec1", "rec2"],
    )

    weights = xr.DataArray(
        np.ones(32, dtype=np.float64).reshape(1, 2, 4, 2, 2),
        dims=["time", "antenna", "frequency", "rec1", "rec2"],
    )

    gaintable = xr.Dataset(
        {
            "gain": gains,
            "weight": weights,
        },
        coords=coords,
        attrs={"configuration": "Antenna Configuration"},
    )

    delay = calculate_delay(gaintable, 4)

    actual_gaintable = apply_delay(gaintable, delay)

    expected_gain = np.array(
        [
            [
                [
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                ],
                [
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                    [
                        [1.0 + 1.48741681e-17j, -0.5 + 8.66025404e-01j],
                        [0.5 + 8.66025404e-01j, 1.0 - 2.57628149e-17j],
                    ],
                ],
            ]
        ]
    )

    np.testing.assert_allclose(
        np.angle(actual_gaintable.gain.data, deg=True),
        np.angle(expected_gain, deg=True),
    )


@patch("ska_sdp_instrumental_calibration.xarray_processors.delay.run_solver")
def test_should_calibrate_single_polarization(run_solver_mock):
    pol = "XX"
    vis_mock = Mock(name="vis")
    modelvis_mock = Mock(name="modelvis")
    initialtable_mock = Mock(name="initialtable")
    solver_mock = Mock(name="solver")

    scalar_vis_mock = Mock(name="scalar_vis")
    scalar_modelvis_mock = Mock(name="scalar_modelvis")
    scalar_table_mock = Mock(name="scalar_table")
    solver_result_mock = Mock(name="solver_result")

    vis_mock.sel.return_value = scalar_vis_mock
    modelvis_mock.sel.return_value = scalar_modelvis_mock
    initialtable_mock.sel.return_value = scalar_table_mock
    run_solver_mock.return_value = solver_result_mock

    result = calibrate_polarization(
        pol, vis_mock, modelvis_mock, initialtable_mock, solver_mock
    )

    vis_mock.sel.assert_called_once_with(polarisation=["XX"])
    modelvis_mock.sel.assert_called_once_with(polarisation=["XX"])
    initialtable_mock.sel.assert_called_once_with(
        receptor1=["X"], receptor2=["X"]
    )
    run_solver_mock.assert_called_once_with(
        vis=scalar_vis_mock,
        modelvis=scalar_modelvis_mock,
        gaintable=scalar_table_mock,
        solver=solver_mock,
    )
    assert result is solver_result_mock


def test_should_stack_jones_coordinate():
    gain_data = np.ones((1, 2, 3, 2, 2), dtype=complex)
    gaintable = xr.Dataset(
        {
            "gain": xr.DataArray(
                gain_data,
                dims=[
                    "time",
                    "antenna",
                    "frequency",
                    "receptor1",
                    "receptor2",
                ],
                coords={"receptor1": ["X", "Y"], "receptor2": ["X", "Y"]},
            )
        }
    )

    result = stack_jones_coordinate(gaintable)

    assert "Jones_Solutions" in result.dims
    assert "receptor1" not in result.dims
    assert "receptor2" not in result.dims
    np.testing.assert_array_equal(
        result["Jones_Solutions"].values, ["J_XX", "J_XY", "J_YX", "J_YY"]
    )


def test_should_unstack_jones_coordinate():
    ref_gain_data = np.zeros((1, 1, 4, 2, 2), dtype=complex)
    ref_gaintable = xr.Dataset(
        {
            "gain": xr.DataArray(
                ref_gain_data,
                dims=[
                    "time",
                    "antenna",
                    "frequency",
                    "receptor1",
                    "receptor2",
                ],
            )
        },
        coords={
            "antenna": [0],
            "frequency": np.linspace(100e6, 150e6, 4),
            "time": [0.0],
        },
    )

    # stacked gain: index 0 placed on [0,0] diagonal, index 1 on [1,1] diagonal
    xx_values = np.array([1 + 0j, 3 + 0j, 5 + 0j, 7 + 0j])
    yy_values = np.array([2 + 0j, 4 + 0j, 6 + 0j, 8 + 0j])
    stacked_gain_data = np.stack([xx_values, yy_values], axis=-1).reshape(
        1, 1, 4, 2
    )
    stacked_gaintable = xr.Dataset(
        {
            "gain": xr.DataArray(
                stacked_gain_data,
                dims=["time", "antenna", "frequency", "Jones_Solutions"],
            )
        }
    )

    result = unstack_jones_coordinate(ref_gaintable, stacked_gaintable)

    np.testing.assert_array_equal(result.gain.data[0, 0, :, 0, 0], xx_values)
    np.testing.assert_array_equal(result.gain.data[0, 0, :, 1, 1], yy_values)
    np.testing.assert_array_equal(result.gain.data[0, 0, :, 0, 1], np.zeros(4))
    np.testing.assert_array_equal(result.gain.data[0, 0, :, 1, 0], np.zeros(4))
