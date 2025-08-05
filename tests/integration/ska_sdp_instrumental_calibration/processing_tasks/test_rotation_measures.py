import dask.array as da
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
            dtype=np.float32,
        ),
    }
    gain_data = (
        np.arange(32, dtype=np.float32)
        + 1
        + 1j * (np.arange(32, dtype=np.float32) + 1)
    ).reshape(1, 2, 4, 2, 2)
    gains = da.from_array(gain_data, chunks=(1, 2, 4, 2, 2))
    weight_data = np.ones_like(gain_data, dtype=np.float32)
    weight = da.from_array(weight_data, chunks=(1, 2, 4, 2, 2))
    gaintable = xr.Dataset(
        {
            "gain": (["time", "antenna", "frequency", "rec1", "rec2"], gains),
            "weight": (
                ["time", "antenna", "frequency", "rec1", "rec2"],
                weight,
            ),
        },
        coords=coords,
    )
    actual_rotations = model_rotations(
        gaintable, refine_fit=True, refant=0, oversample=99
    )

    actual_rm_est_computed = actual_rotations.rm_est.compute()
    expected_rm_est = np.array([-3.662733e-314, -9.485889e001])

    np.testing.assert_allclose(
        actual_rm_est_computed, expected_rm_est, rtol=1e-5  # , atol=1e-10
    )
