import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.processing_tasks.rotation_measures import (  # noqa: E501
    model_rotations,
)


def test_model_rotations():
    coords = {
        "antenna": ["antenna1", "antenna2"],
        "frequency": np.array(
            [1.001350e08, 1.001404e08, 1.001458e08, 1.001512e08],
            dtype=np.float64,
        ),
    }

    gains = xr.DataArray(
        np.arange(32, dtype=np.float64).reshape(1, 2, 4, 2, 2),
        dims=["time", "antenna", "frequency", "rec1", "rec2"],
    )
    gaintable = xr.Dataset(
        {
            "gain": gains,
        },
        coords=coords,
    )

    actual_gaintable = model_rotations(gaintable)

    expected_gain = np.array(
        [
            [
                [
                    [[1.0, -0.0], [0.0, 1.0]],
                    [[1.0, -0.0], [0.0, 1.0]],
                    [[1.0, -0.0], [0.0, 1.0]],
                    [[1.0, -0.0], [0.0, 1.0]],
                ],
                [
                    [[0.31254094, -0.94990429], [0.94990429, 0.31254094]],
                    [[0.18346674, -0.98302592], [0.98302592, 0.18346674]],
                    [[0.05115623, -0.99869066], [0.99869066, 0.05115623]],
                    [[-0.08204089, -0.99662896], [0.99662896, -0.08204089]],
                ],
            ]
        ]
    )

    np.testing.assert_allclose(actual_gaintable.gain, expected_gain)
