# pylint: skip-file
# flake8: noqa

import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.xarray_processors.solver import (
    _run_solver_map_block_,
    run_solver,
)


def test_should_raise_error_for_invalid_frequency_dimension():
    vis_mock = MagicMock(name="vis")
    modelvis_mock = MagicMock(name="modelvis")
    gaintable_mock = MagicMock(name="gaintable")
    solver_mock = MagicMock(name="Solver")

    gaintable_mock.rename.return_value = gaintable_mock
    gaintable_mock.jones_type = "G"
    gaintable_mock.frequency.size = 2

    with pytest.raises(AssertionError):
        run_solver(vis_mock, modelvis_mock, gaintable_mock, solver_mock)


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors.solver.xr.map_blocks"
)
@patch("ska_sdp_instrumental_calibration.xarray_processors.solver.xr.concat")
def test_should_calculate_gaintable_from_visibility_and_model_vis_b_type(
    concat_mock, map_block_mock
):
    vis_mock = MagicMock(name="vis")
    vis_mock.isel.return_value = vis_mock
    modelvis_mock = MagicMock(name="modelvis")
    modelvis_mock.isel.return_value = modelvis_mock
    gaintable_mock = MagicMock(name="gaintable")
    solver_mock = MagicMock(name="Solver")

    combined_gaintable_mock = MagicMock(name="combined_gaintable")
    concat_mock.return_value = combined_gaintable_mock
    combined_gaintable_mock.dims = []
    combined_gaintable_mock.rename.return_value = combined_gaintable_mock

    gaintable_mock.rename.return_value = gaintable_mock
    gaintable_mock.jones_type = "B"
    gaintable_mock.chunk.return_value = gaintable_mock
    gaintable_mock.soln_interval_slices = [1, 2]
    gaintable_mock.frequency.size = 2

    expected = run_solver(vis_mock, modelvis_mock, gaintable_mock, solver_mock)

    map_block_mock.assert_has_calls(
        [
            call(
                _run_solver_map_block_,
                vis_mock.chunk.return_value,
                args=[
                    modelvis_mock.chunk.return_value,
                    gaintable_mock.isel.return_value,
                ],
                kwargs={
                    "solver": solver_mock,
                },
                template=gaintable_mock.isel.return_value,
            ),
            call(
                _run_solver_map_block_,
                vis_mock.chunk.return_value,
                args=[
                    modelvis_mock.chunk.return_value,
                    gaintable_mock.isel.return_value,
                ],
                kwargs={
                    "solver": solver_mock,
                },
                template=gaintable_mock.isel.return_value,
            ),
        ]
    )

    gaintable_mock.rename.assert_called_once_with(time="solution_time")
    gaintable_mock.isel.assert_has_calls(
        [call(solution_time=[0]), call(solution_time=[1])]
    )
    vis_mock.isel.assert_has_calls([call(time=1), call(time=2)])
    modelvis_mock.isel.assert_has_calls([call(time=1), call(time=2)])

    vis_mock.chunk.assert_has_calls([call({"time": -1}), call({"time": -1})])
    modelvis_mock.chunk.assert_has_calls(
        [call({"time": -1}), call({"time": -1})]
    )

    concat_mock.assert_called_once_with(
        [map_block_mock.return_value, map_block_mock.return_value],
        dim="solution_time",
    )
    combined_gaintable_mock.rename.assert_called_once_with(
        solution_time="time"
    )

    solver_mock.normalise_gains.assert_called_once_with(
        combined_gaintable_mock.gain
    )
    combined_gaintable_mock.assign.assert_called_once_with(
        {"gain": solver_mock.normalise_gains.return_value}
    )

    assert expected == combined_gaintable_mock.assign.return_value


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors.solver.xr.map_blocks"
)
@patch("ska_sdp_instrumental_calibration.xarray_processors.solver.xr.concat")
def test_should_calculate_gaintable_from_visibility_and_model_vis_non_b_type(
    concat_mock, map_block_mock
):
    vis_mock = MagicMock(name="vis")
    vis_mock.isel.return_value = vis_mock
    modelvis_mock = MagicMock(name="modelvis")
    modelvis_mock.isel.return_value = modelvis_mock
    gaintable_mock = MagicMock(name="gaintable")
    solver_mock = MagicMock(name="Solver")

    combined_gaintable_mock = MagicMock(name="combined_gaintable")
    concat_mock.return_value = combined_gaintable_mock
    combined_gaintable_mock.dims = ["solution_frequency"]
    combined_gaintable_mock.rename.return_value = combined_gaintable_mock

    gaintable_mock.rename.return_value = gaintable_mock
    gaintable_mock.jones_type = "G"
    gaintable_mock.chunk.return_value = gaintable_mock
    gaintable_mock.soln_interval_slices = [1]
    gaintable_mock.frequency.size = 1

    expected = run_solver(vis_mock, modelvis_mock, gaintable_mock, solver_mock)

    map_block_mock.assert_called_once_with(
        _run_solver_map_block_,
        vis_mock.chunk.return_value,
        args=[
            modelvis_mock.chunk.return_value,
            gaintable_mock.isel.return_value,
        ],
        kwargs={
            "solver": solver_mock,
        },
        template=gaintable_mock.isel.return_value,
    )

    gaintable_mock.rename.assert_has_calls(
        [call(time="solution_time"), call(frequency="solution_frequency")]
    )
    gaintable_mock.isel.assert_called_once_with(solution_time=[0])
    vis_mock.isel.assert_called_once_with(time=1)
    modelvis_mock.isel.assert_called_once_with(time=1)

    vis_mock.chunk.assert_called_once_with({"time": -1, "frequency": -1})
    modelvis_mock.chunk.assert_called_once_with({"time": -1, "frequency": -1})

    concat_mock.assert_called_once_with(
        [map_block_mock.return_value],
        dim="solution_time",
    )
    combined_gaintable_mock.rename.assert_has_calls(
        [call(solution_time="time"), call(solution_frequency="frequency")]
    )

    solver_mock.normalise_gains.assert_called_once_with(
        combined_gaintable_mock.gain
    )
    combined_gaintable_mock.assign.assert_called_once_with(
        {"gain": solver_mock.normalise_gains.return_value}
    )

    assert expected == combined_gaintable_mock.assign.return_value


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".solver.restore_baselines_dim"
)
def test_should_run_solver_map_block(restore_dim_mock):
    vis_mock = MagicMock(name="vis")
    modelvis_mock = MagicMock(name="modelvis")
    gaintable_mock = MagicMock(name="gaintable")
    solver = MagicMock(name="solver")

    restore_dim_mock.side_effect = [vis_mock, modelvis_mock]

    solver.solve.return_value = ["gain", "weight", "residual"]

    expected_gaintable = _run_solver_map_block_(
        vis_mock, modelvis_mock, gaintable_mock, solver
    )

    restore_dim_mock.assert_has_calls([call(vis_mock), call(modelvis_mock)])

    solver.solve.assert_called_once_with(
        vis_mock.vis.data,
        vis_mock.flags.data,
        vis_mock.weight.data,
        modelvis_mock.vis.data,
        modelvis_mock.flags.data,
        gaintable_mock.gain.data,
        gaintable_mock.weight.data,
        gaintable_mock.residual.data,
        vis_mock.antenna1.data,
        vis_mock.antenna2.data,
    )

    assert expected_gaintable.gain.data == "gain"
    assert expected_gaintable.weight.data == "weight"
    assert expected_gaintable.residual.data == "residual"
