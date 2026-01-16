import dask.array as da
import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors.rotation_measures import (  # noqa: E501
    ModelRotationData,
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
    expected_rm_est = np.array([0, -94.9161247])

    np.testing.assert_allclose(
        actual_rm_est_computed, expected_rm_est, atol=1e-7
    )


def test_should_return_plot_params_for_station():

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

    rot_data = ModelRotationData(gaintable, refant=0)

    rot_data.rm_spec = [1, 2]

    plot_params = rot_data.get_plot_params_for_station()
    stn = len(gaintable.antenna) - 1

    assert "J" in plot_params
    assert "lambda_sq" in plot_params
    assert "xlim" in plot_params

    assert all(plot_params["rm_vals"] == rot_data.rm_vals)
    assert plot_params["rm_spec"] == rot_data.rm_spec[stn]
    assert plot_params["rm_peak"] == rot_data.rm_peak[stn]
    assert plot_params["rm_est"] == rot_data.rm_est[stn]
    assert plot_params["rm_est_refant"] == rot_data.rm_est[rot_data.refant]
    assert plot_params["stn"] == stn

    rot_data = ModelRotationData(gaintable, refant=0)

    rot_data.rm_spec = [1, 2]

    plot_params = rot_data.get_plot_params_for_station(1)
    stn = 1

    assert all(plot_params["rm_vals"] == rot_data.rm_vals)
    assert plot_params["rm_spec"] == rot_data.rm_spec[stn]
    assert plot_params["rm_peak"] == rot_data.rm_peak[stn]
    assert plot_params["rm_est"] == rot_data.rm_est[stn]
    assert plot_params["rm_est_refant"] == rot_data.rm_est[rot_data.refant]
    assert plot_params["stn"] == stn
