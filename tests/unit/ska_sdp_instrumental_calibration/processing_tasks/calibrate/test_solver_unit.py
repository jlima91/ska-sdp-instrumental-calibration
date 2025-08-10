# flake8: noqa

import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.processing_tasks.calibrate.solver import (
    _solve_gaintable,
    run_solver,
)


def test_should_raise_error_if_gaintable_time_size_is_not_one():
    vis = MagicMock(name="vis")
    modelvis = MagicMock(name="modelvis")

    gaintable = MagicMock(name="gaintable")
    gaintable.time = np.array([1, 2])

    with pytest.raises(ValueError) as err:
        run_solver(vis, modelvis, gaintable=gaintable)

    assert (
        str(err.value)
        == "Error setting up gaintable. Size of 'time' dimension is not 1."
    )


def test_should_raise_error_if_reference_antenna_is_invalid():
    vis = MagicMock(name="vis")
    modelvis = MagicMock(name="modelvis")

    gaintable = MagicMock(name="gaintable")
    gaintable.time = np.array([1])
    gaintable.antenna = np.array([0, 1, 2])

    with pytest.raises(ValueError) as err:
        run_solver(vis, modelvis, gaintable=gaintable, refant=-1)

    assert str(err.value) == "Invalid refant: -1"

    with pytest.raises(ValueError) as err:
        run_solver(vis, modelvis, gaintable=gaintable, refant=5)

    assert str(err.value) == "Invalid refant: 5"


def test_should_raise_error_for_invalid_frequency_dimension():
    vis = MagicMock(name="vis")
    vis.frequency = xr.DataArray(np.array([0, 1, 2]), dims=["frequency"])
    modelvis = MagicMock(name="modelvis")
    modelvis.frequency = xr.DataArray(np.array([0, 1, 2]), dims=["frequency"])

    gaintable = MagicMock(name="gaintable")
    gaintable.time = np.array([1])
    gaintable.antenna = np.array([0, 1, 2, 3, 4])
    gaintable.frequency = xr.DataArray(np.array([0, 1]), dims=["frequency"])

    with pytest.raises(ValueError) as err:
        run_solver(vis, modelvis, gaintable=gaintable, refant=1)

    assert (
        str(err.value) == "Only supports single-channel or all-channel output"
    )

    gaintable.frequency = xr.DataArray(np.array([10]), dims=["frequency"])

    with pytest.raises(ValueError) as err:
        run_solver(vis, modelvis, gaintable=gaintable, refant=1)

    assert str(err.value) == "Single-channel output is at the wrong frequency"


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.calibrate.solver.restore_baselines_dim"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.calibrate.solver.create_bandpass_table"
)
def test_should_calculate_gaintable_from_visibility_and_model_vis(
    create_bandpass_table_mock, restore_baselines_dim_mock
):
    # Arrange
    vis = MagicMock(name="vis")
    vis.frequency = xr.DataArray(np.array([0, 1, 2]), dims=["frequency"])
    modelvis = MagicMock(name="modelvis")
    modelvis.frequency = xr.DataArray(np.array([0, 1, 2]), dims=["frequency"])

    gaintable = MagicMock(name="gaintable")
    gaintable.time = np.array([1])
    gaintable.antenna = np.array([0, 1, 2, 3, 4])
    # Mean of vis frequency, should set "jones_type" to "G"
    gaintable.frequency = xr.DataArray(np.array([1]), dims=["frequency"])
    time_renamed_gaintable = gaintable.rename.return_value
    map_blocked_gaintable = time_renamed_gaintable.map_blocks.return_value

    restore_baselines_dim_mock.side_effect = [
        "restored_vis",
        "restored_modelvis",
    ]

    # Act
    result = run_solver(vis, modelvis, gaintable=gaintable, refant=1)

    # Asserts
    gaintable.rename.assert_called_once_with({"time": "soln_time"})

    restore_baselines_dim_mock.assert_has_calls([call(vis), call(modelvis)])

    time_renamed_gaintable.map_blocks.assert_called_once_with(
        _solve_gaintable,
        args=[
            "restored_vis",
            "restored_modelvis",
            False,
            200,
            1e-6,
            False,
            None,
            "gain_substitution",
            "G",
            None,
            1,
        ],
        template=time_renamed_gaintable,
    )

    map_blocked_gaintable.rename.assert_called_once_with({"soln_time": "time"})

    assert result is map_blocked_gaintable.rename.return_value
