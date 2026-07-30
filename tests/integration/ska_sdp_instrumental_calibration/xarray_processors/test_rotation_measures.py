import dask.array as da
import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors.rotation_measures import (  # noqa: E501
    RotationMeasureData,
    model_rotations,
    get_plot_params_for_station
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
            "gain": (["time", "antenna", "frequency", "receptor1", "receptor2"], gains),
            "weight": (
                ["time", "antenna", "frequency", "receptor1", "receptor2"],
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
            "gain": (["time", "antenna", "frequency", "receptor1", "receptor2"], gains),
            "weight": (
                ["time", "antenna", "frequency", "receptor1", "receptor2"],
                weight,
            ),
        },
        coords=coords,
    )

    rot_data = model_rotations(gaintable)

    # rot_data.rm_spec = [1, 2]

    stn = len(gaintable.antenna) - 1
    plot_params = get_plot_params_for_station(rot_data,stn,0)

    # raise Exception(plot_params)

    assert "J" in plot_params
    assert "lambda_sq" in plot_params
    assert "xlim" in plot_params

    assert all(plot_params["rm_vals"] == rot_data.resolution)
    assert all(plot_params["rm_spec"] == rot_data.rm_spec[0, stn])
    assert plot_params["rm_peak"] == rot_data.rm_peak[0, stn]
    assert plot_params["rm_est"] == rot_data.rm_est[0, stn]
    assert plot_params["rm_est_refant"] == rot_data.rm_est[0, 0]
    assert plot_params["stn"] == stn

    rot_data = model_rotations(gaintable)

    # rot_data.rm_spec = [1, 2]

    plot_params = get_plot_params_for_station(rot_data, 1, 0)
    stn = 1

    assert all(plot_params["rm_vals"] == rot_data.resolution)
    assert all(plot_params["rm_spec"] == rot_data.rm_spec[0, stn])
    assert plot_params["rm_peak"] == rot_data.rm_peak[0, stn]
    assert plot_params["rm_est"] == rot_data.rm_est[0, stn]
    assert plot_params["rm_est_refant"] == rot_data.rm_est[0, 0]
    assert plot_params["stn"] == stn
