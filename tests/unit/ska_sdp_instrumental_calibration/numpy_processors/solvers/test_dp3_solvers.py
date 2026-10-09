import os
import sys

import numpy as np
import pytest
from mock import ANY, MagicMock, call, patch
from ska_sdp_func_python.visibility.operations import expand_polarizations

from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_solvers import (  # noqa: E501
    Dp3GaincalSolver,
    DP3ObservationInfo,
    _create_dp_info,
    _get_dp3_caltype,
    _n_per_interval,
    dp3_gaincal_solve,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers.solver import (
    Solver,
)

_MODULE = (
    "ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_solvers"
)

_N_TIME = 4
_N_BL = 6
_N_FREQ = 2
_N_ANT = 3
_ANTENNA_NAMES = ["ANT0", "ANT1", "ANT2"]
_ANT1 = np.array([0, 0, 1])
_ANT2 = np.array([1, 2, 2])


@pytest.fixture
def dp3_unavailable():
    with patch(
        "ska_sdp_instrumental_calibration.numpy_processors.solvers."
        "dp3_threading.is_dp3_available",
        return_value=False,
    ):
        yield


@pytest.fixture
def vis_data():
    """Observed and model visibilities with flags and weights."""
    shape = (_N_TIME, _N_BL, _N_FREQ, 4)
    rng = np.random.default_rng(42)
    vis = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    model = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    flags = np.zeros(shape, dtype=bool)
    flags[0, 0, 0, :] = True
    weight = rng.uniform(size=shape).astype(np.float32)
    return vis, flags, weight, model


@pytest.fixture
def gain_data():
    """Initial gains, weights and residuals for 2 solution times."""
    shape = (2, _N_ANT, 1, 2, 2)
    gain = np.broadcast_to(np.eye(2, dtype=np.complex64), shape).copy()
    weight = np.ones(shape, dtype=np.float32)
    residual = np.zeros((2, 1, 2, 2), dtype=np.float32)
    return gain, weight, residual


@pytest.fixture
def observation():
    """Observation metadata matching the ``vis_data`` fixture."""
    return DP3ObservationInfo(
        antenna_names=_ANTENNA_NAMES,
        antenna_positions=np.arange(_N_ANT * 3, dtype=float).reshape(
            _N_ANT, 3
        ),
        antenna_diameters=np.full(_N_ANT, 38.0),
        frequency=np.array([100e6, 101e6]),
        channel_bandwidth=np.array([1e6, 1e6]),
        time=np.arange(_N_TIME) * 10.0 + 5e9,
        integration_time=10.0,
        uvw=np.arange(_N_TIME * _N_BL * 3, dtype=float).reshape(
            _N_TIME, _N_BL, 3
        ),
        phasecentre=(0.1, -0.5),
    )


@pytest.fixture
def mock_dp3(monkeypatch, gain_data):
    """
    Replace the ``dp3`` package and the helpers of ``dp3_gaincal_solve``
    with mocks.

    All mocks are children of a single ``manager`` mock, so the order of
    the calls across them can be asserted.
    """
    manager = MagicMock(name="manager")

    dp3 = manager.dp3
    parameterset = MagicMock(name="dp3.parameterset")
    dp3.parameterset = parameterset
    monkeypatch.setitem(sys.modules, "dp3", dp3)
    monkeypatch.setitem(sys.modules, "dp3.parameterset", parameterset)

    parsets = [MagicMock(name="parset"), MagicMock(name="null_parset")]
    parameterset.ParameterSet.side_effect = parsets

    gaincal_step = manager.gaincal_step
    null_step = MagicMock(name="null_step")
    dp3.make_step.side_effect = [gaincal_step, null_step]

    buffers = []

    def _make_buffer():
        buffer = MagicMock(name=f"buffer{len(buffers)}")
        buffers.append(buffer)
        return buffer

    dp3.DPBuffer.side_effect = _make_buffer

    # Fake solutions marking only the diagonal of the first antenna as
    # solved, so that the merge with the input gains can be checked
    gain, _, _ = gain_data
    dp3_gain = np.full(gain.shape, 2.0 + 1.0j)
    dp3_weight = np.full(gain.shape, 0.5)
    solved = np.zeros(gain.shape, dtype=bool)
    solved[:, 0, :, 0, 0] = True
    solved[:, 0, :, 1, 1] = True
    manager.read_h5parm_gains.return_value = (
        dp3_gain,
        dp3_weight,
        solved,
        None,
    )

    manager.default_dp3_n_threads.return_value = 8

    with patch(f"{_MODULE}.require_dp3", manager.require_dp3), patch(
        f"{_MODULE}.get_dp3_lock", manager.get_dp3_lock
    ), patch(
        f"{_MODULE}.default_dp3_n_threads", manager.default_dp3_n_threads
    ), patch(
        f"{_MODULE}.read_h5parm_gains", manager.read_h5parm_gains
    ), patch(
        f"{_MODULE}._create_dp_info", manager.create_dp_info
    ):
        yield {
            "manager": manager,
            "dp3": dp3,
            "parset": parsets[0],
            "null_parset": parsets[1],
            "gaincal_step": gaincal_step,
            "null_step": null_step,
            "buffers": buffers,
            "lock": manager.get_dp3_lock.return_value,
            "dp3_gain": dp3_gain,
            "dp3_weight": dp3_weight,
            "solved": solved,
        }


def _solve(vis_data, gain_data, model_flags=None, **kwargs):
    """Call ``dp3_gaincal_solve`` with the fixture data."""
    vis, flags, weight, model = vis_data
    gain, gain_weight, residual = gain_data
    if "model_vis" in kwargs:
        model = kwargs.pop("model_vis")
    return dp3_gaincal_solve(
        vis,
        flags,
        weight,
        model,
        model_flags,
        gain,
        gain_weight,
        residual,
        _ANT1,
        _ANT2,
        **kwargs,
    )


@patch(f"{_MODULE}.require_dp3")
def test_dp3_gaincal_solver_should_require_dp3(require_dp3_mock):
    Dp3GaincalSolver()

    require_dp3_mock.assert_called_once_with("dp3_gaincal solver")


def test_dp3_gaincal_solver_should_raise_if_dp3_unavailable(dp3_unavailable):
    with pytest.raises(
        ImportError,
        match="dp3_gaincal solver requires the optional 'dp3' package",
    ):
        Dp3GaincalSolver()


@patch(f"{_MODULE}.require_dp3")
def test_dp3_gaincal_solver_should_set_defaults(_require_dp3_mock):
    solver = Dp3GaincalSolver()

    assert solver.crosspol is False
    assert solver.phase_only is False
    assert solver.caltype == "diagonal"
    assert solver.niter == 50
    assert solver.tol == 1e-6


@patch(f"{_MODULE}.require_dp3")
def test_dp3_gaincal_solver_should_be_registered(_require_dp3_mock):
    solver = Solver.get_solver(
        "dp3_gaincal", crosspol=True, niter=10, tol=1e-3, unknown=1
    )

    assert isinstance(solver, Dp3GaincalSolver)
    assert solver.crosspol is True
    assert solver.phase_only is False
    assert solver.caltype == "fulljones"
    assert solver.niter == 10
    assert solver.tol == 1e-3


@patch(f"{_MODULE}.require_dp3")
def test_dp3_gaincal_solver_should_raise_for_crosspol_with_phase_only(
    _require_dp3_mock,
):
    with pytest.raises(
        ValueError,
        match="phase_only can not be combined with crosspol",
    ):
        Dp3GaincalSolver(crosspol=True, phase_only=True)


@patch(f"{_MODULE}.dp3_gaincal_solve")
@patch(f"{_MODULE}.require_dp3")
def test_dp3_gaincal_solver_solve_should_raise_without_model(
    _require_dp3_mock, dp3_gaincal_solve_mock
):
    solver = Dp3GaincalSolver()

    with pytest.raises(
        ValueError, match="dp3_gaincal: model_vis must be provided"
    ):
        solver.solve(*["arg"] * 3, None, *["arg"] * 6)

    dp3_gaincal_solve_mock.assert_not_called()


@patch(f"{_MODULE}.dp3_gaincal_solve")
@patch(f"{_MODULE}.require_dp3")
def test_dp3_gaincal_solver_solve_should_call_dp3_gaincal_solve(
    _require_dp3_mock, dp3_gaincal_solve_mock
):
    solver = Dp3GaincalSolver(phase_only=True, niter=10, tol=1e-3)
    args = [MagicMock(name=f"arg{idx}") for idx in range(10)]

    result = solver.solve(*args)

    dp3_gaincal_solve_mock.assert_called_once_with(
        *args, caltype="diagonalphase", niter=10, tol=1e-3
    )
    assert result == dp3_gaincal_solve_mock.return_value


def test_dp3_observation_info_should_be_created_from_visibility(
    generate_vis,
):
    vis, _ = generate_vis

    observation = DP3ObservationInfo.from_visibility(vis)

    assert observation.antenna_names == list(vis.configuration.names.data)
    np.testing.assert_array_equal(
        observation.antenna_positions, vis.configuration.xyz.data
    )
    np.testing.assert_array_equal(
        observation.antenna_diameters, vis.configuration.diameter.data
    )
    np.testing.assert_array_equal(observation.frequency, vis.frequency.data)
    np.testing.assert_array_equal(
        observation.channel_bandwidth, vis.channel_bandwidth.data
    )
    np.testing.assert_array_equal(observation.time, vis.time.data)
    assert observation.integration_time == vis.integration_time.data[0]
    assert isinstance(observation.integration_time, float)
    np.testing.assert_array_equal(observation.uvw, vis.uvw.data)
    assert observation.phasecentre == (
        vis.phasecentre.ra.rad,
        vis.phasecentre.dec.rad,
    )


def test_dp3_gaincal_solve_should_raise_if_dp3_unavailable(
    dp3_unavailable, vis_data, gain_data
):
    with pytest.raises(
        ImportError,
        match="dp3_gaincal_solve requires the optional 'dp3' package",
    ):
        _solve(vis_data, gain_data)


@pytest.mark.parametrize(
    "with_skymodel,with_observation",
    [(False, False), (True, False), (False, True)],
    ids=["neither", "skymodel_only", "observation_only"],
)
def test_dp3_gaincal_solve_should_raise_without_model_or_skymodel(
    mock_dp3,
    vis_data,
    gain_data,
    observation,
    with_skymodel,
    with_observation,
):
    with pytest.raises(
        ValueError,
        match="skymodel_path and observation are required when model_vis "
        "is not provided",
    ):
        _solve(
            vis_data,
            gain_data,
            model_vis=None,
            skymodel_path="sky.txt" if with_skymodel else None,
            observation=observation if with_observation else None,
        )

    mock_dp3["dp3"].make_step.assert_not_called()


def test_dp3_gaincal_solve_should_raise_for_unsupported_caltype(
    mock_dp3, vis_data, gain_data
):
    with pytest.raises(
        ValueError,
        match="dp3_gaincal: unsupported caltype 'tec'. Supported: "
        "scalarphase, diagonal, diagonalamplitude, diagonalphase, fulljones",
    ):
        _solve(vis_data, gain_data, caltype="tec")

    mock_dp3["dp3"].make_step.assert_not_called()


@pytest.mark.parametrize(
    "gain_shape,axis,n_samples,n_intervals",
    [
        ((3, _N_ANT, 1, 2, 2), "time", _N_TIME, 3),
        ((2, _N_ANT, 3, 2, 2), "frequency", _N_FREQ, 3),
    ],
    ids=["time", "frequency"],
)
def test_dp3_gaincal_solve_should_raise_for_incompatible_gain_shape(
    mock_dp3, vis_data, gain_shape, axis, n_samples, n_intervals
):
    gain_data = (
        np.ones(gain_shape, dtype=complex),
        np.ones(gain_shape),
        np.zeros(gain_shape[:1] + gain_shape[2:]),
    )

    with pytest.raises(
        ValueError,
        match=f"cannot split {n_samples} {axis} samples into "
        f"{n_intervals} solution intervals",
    ):
        _solve(vis_data, gain_data)

    mock_dp3["dp3"].make_step.assert_not_called()


def test_dp3_gaincal_solve_should_configure_gaincal_with_model(
    mock_dp3, vis_data, gain_data
):
    dp3 = mock_dp3["dp3"]

    _solve(vis_data, gain_data, caltype="fulljones", niter=10, tol=1e-3)

    mock_dp3["manager"].require_dp3.assert_called_once_with(
        "dp3_gaincal_solve"
    )
    mock_dp3["parset"].add.assert_has_calls(
        [
            call("gaincal.parmdb", ANY),
            call("gaincal.reusemodel", "modeldata"),
            call("gaincal.caltype", "fulljones"),
            call("gaincal.solint", "2"),
            call("gaincal.nchan", "2"),
            call("gaincal.maxiter", "10"),
            call("gaincal.tolerance", "0.001"),
        ]
    )
    assert mock_dp3["parset"].add.call_count == 7
    dp3.make_step.assert_has_calls(
        [
            call(
                "gaincal",
                mock_dp3["parset"],
                "gaincal.",
                dp3.MsType.regular,
            ),
            call("null", mock_dp3["null_parset"], "", dp3.MsType.regular),
        ]
    )
    mock_dp3["gaincal_step"].set_next_step.assert_called_once_with(
        mock_dp3["null_step"]
    )
    mock_dp3["manager"].create_dp_info.assert_called_once_with(
        _ANT1, _ANT2, _N_ANT, _N_TIME, _N_FREQ, None
    )
    mock_dp3["gaincal_step"].set_info.assert_called_once_with(
        mock_dp3["manager"].create_dp_info.return_value
    )


def test_dp3_gaincal_solve_should_read_solutions_from_temporary_h5parm(
    mock_dp3, vis_data, gain_data
):
    _solve(vis_data, gain_data)

    h5parm_path = mock_dp3["parset"].add.call_args_list[0].args[1]
    assert os.path.basename(h5parm_path) == "gaincal.h5"
    mock_dp3["manager"].read_h5parm_gains.assert_called_once_with(h5parm_path)
    # The temporary directory is removed after reading the solutions
    assert not os.path.exists(os.path.dirname(h5parm_path))


def test_dp3_gaincal_solve_should_stream_timeslots_with_model(
    mock_dp3, vis_data, gain_data
):
    vis, flags, weight, model = vis_data

    _solve(vis_data, gain_data)

    buffers = mock_dp3["buffers"]
    assert len(buffers) == _N_TIME
    mock_dp3["gaincal_step"].process.assert_has_calls(
        [call(buffer) for buffer in buffers]
    )
    for time_idx, buffer in enumerate(buffers):
        buffer.set_time.assert_called_once_with(time_idx + 0.5)
        np.testing.assert_array_equal(
            buffer.set_uvw.call_args.args[0], np.zeros((_N_BL, 3))
        )
        np.testing.assert_array_equal(
            buffer.set_data.call_args.args[0],
            expand_polarizations(vis[time_idx], np.complex64),
        )
        np.testing.assert_array_equal(
            buffer.set_weights.call_args.args[0],
            expand_polarizations(weight[time_idx], np.float32),
        )
        np.testing.assert_array_equal(
            buffer.set_flags.call_args.args[0],
            expand_polarizations(flags[time_idx], bool),
        )
        buffer.add_data.assert_called_once_with("modeldata")
        name, extra_data = buffer.set_extra_data.call_args.args
        assert name == "modeldata"
        np.testing.assert_array_equal(
            extra_data, expand_polarizations(model[time_idx], np.complex64)
        )
    mock_dp3["gaincal_step"].finish.assert_called_once_with()


def test_dp3_gaincal_solve_should_combine_vis_and_model_flags(
    mock_dp3, vis_data, gain_data
):
    _, flags, _, _ = vis_data
    model_flags = np.zeros(flags.shape, dtype=bool)
    model_flags[1, 2, 1, :] = True

    _solve(vis_data, gain_data, model_flags=model_flags)

    for time_idx, buffer in enumerate(mock_dp3["buffers"]):
        np.testing.assert_array_equal(
            buffer.set_flags.call_args.args[0],
            expand_polarizations(
                flags[time_idx] | model_flags[time_idx], bool
            ),
        )


def test_dp3_gaincal_solve_should_predict_model_from_skymodel(
    mock_dp3, vis_data, gain_data, observation
):
    _solve(
        vis_data,
        gain_data,
        model_vis=None,
        skymodel_path="sky.txt",
        observation=observation,
    )

    parset_add = mock_dp3["parset"].add
    parset_add.assert_any_call("gaincal.sourcedb", "sky.txt")
    assert "gaincal.reusemodel" not in [
        add_call.args[0] for add_call in parset_add.call_args_list
    ]
    mock_dp3["manager"].create_dp_info.assert_called_once_with(
        _ANT1, _ANT2, _N_ANT, _N_TIME, _N_FREQ, observation
    )
    for time_idx, buffer in enumerate(mock_dp3["buffers"]):
        buffer.set_time.assert_called_once_with(observation.time[time_idx])
        # DP3 uses the opposite uvw sign convention
        np.testing.assert_array_equal(
            buffer.set_uvw.call_args.args[0], -observation.uvw[time_idx]
        )
        buffer.add_data.assert_not_called()
        buffer.set_extra_data.assert_not_called()


def test_dp3_gaincal_solve_should_reuse_model_with_observation(
    mock_dp3, vis_data, gain_data, observation
):
    _solve(
        vis_data,
        gain_data,
        skymodel_path="sky.txt",
        observation=observation,
    )

    parset_add = mock_dp3["parset"].add
    parset_add.assert_any_call("gaincal.reusemodel", "modeldata")
    assert "gaincal.sourcedb" not in [
        add_call.args[0] for add_call in parset_add.call_args_list
    ]
    for time_idx, buffer in enumerate(mock_dp3["buffers"]):
        buffer.set_time.assert_called_once_with(observation.time[time_idx])
        buffer.add_data.assert_called_once_with("modeldata")


def test_dp3_gaincal_solve_should_use_default_number_of_threads(
    mock_dp3, vis_data, gain_data
):
    _solve(vis_data, gain_data)

    mock_dp3["manager"].default_dp3_n_threads.assert_called_once_with()
    mock_dp3["dp3"].set_n_threads.assert_called_once_with(8)


def test_dp3_gaincal_solve_should_use_given_number_of_threads(
    mock_dp3, vis_data, gain_data
):
    _solve(vis_data, gain_data, n_threads=3)

    mock_dp3["manager"].default_dp3_n_threads.assert_not_called()
    mock_dp3["dp3"].set_n_threads.assert_called_once_with(3)


def test_dp3_gaincal_solve_should_run_dp3_while_holding_lock(
    mock_dp3, vis_data, gain_data
):
    _solve(vis_data, gain_data)

    call_names = [name for name, _, _ in mock_dp3["manager"].mock_calls]
    enter_idx = call_names.index("get_dp3_lock().__enter__")
    exit_idx = call_names.index("get_dp3_lock().__exit__")
    dp3_idx = [
        idx
        for idx, name in enumerate(call_names)
        if name.startswith(("dp3.", "gaincal_step."))
    ]

    assert enter_idx < min(dp3_idx)
    assert max(dp3_idx) < exit_idx
    assert call_names.index("gaincal_step.finish") < exit_idx
    assert exit_idx < call_names.index("read_h5parm_gains")


def test_dp3_gaincal_solve_should_merge_solved_terms_into_gains(
    mock_dp3, vis_data, gain_data
):
    gain, gain_weight, residual = gain_data
    solved = mock_dp3["solved"]

    new_gain, new_weight, new_residual = _solve(vis_data, gain_data)

    np.testing.assert_array_equal(
        new_gain[solved], mock_dp3["dp3_gain"][solved]
    )
    np.testing.assert_array_equal(new_gain[~solved], gain[~solved])
    np.testing.assert_array_equal(
        new_weight[solved], mock_dp3["dp3_weight"][solved]
    )
    np.testing.assert_array_equal(new_weight[~solved], gain_weight[~solved])
    assert new_gain.dtype == gain.dtype
    assert new_weight.dtype == gain_weight.dtype
    np.testing.assert_array_equal(new_residual, residual)
    assert new_residual is not residual


@pytest.mark.parametrize(
    "n_samples,n_intervals,expected",
    [(4, 1, 4), (4, 2, 2), (4, 4, 1), (5, 2, 3), (10, 3, 4)],
)
def test_n_per_interval(n_samples, n_intervals, expected):
    assert _n_per_interval(n_samples, n_intervals, "time") == expected


@pytest.mark.parametrize(
    "n_samples,n_intervals",
    [(10, 6), (4, 3), (2, 3)],
)
def test_n_per_interval_should_raise_for_uneven_split(n_samples, n_intervals):
    with pytest.raises(
        ValueError,
        match=f"dp3_gaincal: cannot split {n_samples} frequency samples into "
        f"{n_intervals} solution intervals",
    ):
        _n_per_interval(n_samples, n_intervals, "frequency")


@pytest.mark.requires_dp3
def test_create_dp_info_should_use_placeholder_metadata():
    dpinfo = _create_dp_info(
        _ANT1.astype(float),
        _ANT2.astype(float),
        _N_ANT,
        _N_TIME,
        _N_FREQ,
        None,
    )

    assert dpinfo.n_correlations == 4
    assert dpinfo.channel_frequencies == [1.0, 2.0]
    assert dpinfo.channel_widths == [1.0, 1.0]
    assert dpinfo.antenna_names == _ANTENNA_NAMES
    np.testing.assert_array_equal(
        dpinfo.antenna_positions, np.zeros((_N_ANT, 3))
    )
    assert dpinfo.first_antenna_indices == _ANT1.tolist()
    assert dpinfo.second_antenna_indices == _ANT2.tolist()
    assert dpinfo.first_time == 0.5
    assert dpinfo.last_time == _N_TIME - 0.5
    assert dpinfo.time_interval == 1.0
    assert dpinfo.n_times == _N_TIME
    np.testing.assert_allclose(dpinfo.phase_center, [0.0, 0.0])


@pytest.mark.requires_dp3
def test_create_dp_info_should_use_observation_metadata(generate_vis):
    vis, _ = generate_vis
    observation = DP3ObservationInfo.from_visibility(vis)
    ant1 = vis.antenna1.data
    ant2 = vis.antenna2.data

    dpinfo = _create_dp_info(
        ant1,
        ant2,
        vis.configuration.id.size,
        vis.time.size,
        vis.frequency.size,
        observation,
    )

    assert dpinfo.n_correlations == 4
    np.testing.assert_array_equal(
        dpinfo.channel_frequencies, vis.frequency.data
    )
    np.testing.assert_array_equal(
        dpinfo.channel_widths, vis.channel_bandwidth.data
    )
    assert dpinfo.antenna_names == observation.antenna_names
    np.testing.assert_array_equal(
        dpinfo.antenna_positions, vis.configuration.xyz.data
    )
    assert dpinfo.first_antenna_indices == ant1.tolist()
    assert dpinfo.second_antenna_indices == ant2.tolist()
    assert dpinfo.first_time == vis.time.data[0]
    assert dpinfo.last_time == vis.time.data[-1]
    assert dpinfo.time_interval == observation.integration_time
    assert dpinfo.n_times == vis.time.size
    np.testing.assert_allclose(dpinfo.phase_center, observation.phasecentre)


@pytest.mark.parametrize(
    "crosspol,phase_only,expected",
    [
        (False, False, "diagonal"),
        (False, True, "diagonalphase"),
        (True, False, "fulljones"),
    ],
)
def test_get_dp3_caltype(crosspol, phase_only, expected):
    assert _get_dp3_caltype(crosspol, phase_only) == expected


def test_get_dp3_caltype_should_raise_for_unsupported_combination():
    with pytest.raises(
        ValueError,
        match="dp3_gaincal: phase_only can not be combined with crosspol, "
        "as DP3 gaincal has no phase-only fulljones caltype",
    ):
        _get_dp3_caltype(True, True)
