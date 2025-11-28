# pylint: disable = too-many-function-args
import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors.delay import (
    apply_delay,
    calculate_delay,
    calculate_gain_rot,
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
