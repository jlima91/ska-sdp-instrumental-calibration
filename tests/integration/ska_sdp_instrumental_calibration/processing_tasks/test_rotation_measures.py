import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.processing_tasks.rotation_measures import (  # noqa: E501
    model_rotations,
)


def test_model_rotations():
    coords = {
        "time": [0],
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
    weight = xr.DataArray(
        np.ones((1, 2, 4)),
        dims=["time", "antenna", "frequency"],
    )
    gaintable = xr.Dataset(
        {
            "gain": gains,
            "weight": weight,
        },
        coords=coords,
    )

    model_rotations(gaintable, peak_threshold=0.5, refine_fit=True, refant=0)
