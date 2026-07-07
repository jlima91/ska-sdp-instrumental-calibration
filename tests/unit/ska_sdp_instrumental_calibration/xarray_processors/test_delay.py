# pylint: disable = too-many-function-args
import numpy as np
import xarray as xr
from mock import Mock, patch

from ska_sdp_instrumental_calibration.xarray_processors.delay import (
    apply_delay_to_gaintable,
    calculate_delay,
    calculate_gain_rot,
    calibrate_polarization,
    coarse_delay,
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
    actual = calculate_gain_rot(gains, delay, offset, frequency, inverse=True)

    np.testing.assert_allclose(actual, expected)

    actual = calculate_gain_rot(
        expected, delay, offset, frequency, inverse=False
    )

    np.testing.assert_allclose(actual, gains)


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

    actual_gaintable = apply_delay_to_gaintable(gaintable, delay, inverse=True)

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
