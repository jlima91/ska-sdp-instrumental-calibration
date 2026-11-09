import os
import sys

import numpy as np
import pytest
from mock import ANY, MagicMock, call, patch
from ska_sdp_func_python.visibility.operations import expand_polarizations

from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_ddecal_solvers import (  # noqa: E501
    Dp3DdecalSolver,
    _find_phase_centre_patch,
    _parse_skymodel_angle,
    _read_skymodel_patches,
    _register_model_direction,
    _split_skymodel_line,
    dp3_ddecal_solve,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_solvers import (  # noqa: E501
    DP3ObservationInfo,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers.solver import (
    Solver,
)

_MODULE = (
    "ska_sdp_instrumental_calibration.numpy_processors.solvers."
    "dp3_ddecal_solvers"
)

_N_TIME = 4
_N_BL = 3
_N_FREQ = 2
_N_ANT = 3
_ANT1 = np.array([0, 0, 1])
_ANT2 = np.array([1, 2, 2])

_SKYMODEL_WITH_PATCHES = """\
FORMAT = Name, Type, Patch, Ra, Dec, I, SpectralIndex='[]'
,,p_main,10deg,-20deg
,,p_other,11deg,-21deg
SRC_A,POINT,p_main,10deg,-20deg,5.0,[-0.7, 0.1]
SRC_B,POINT,p_main,10.1deg,-20deg,1.0,[-0.7]
SRC_C,POINT,p_other,11deg,-21deg,1.0,[-0.7]
"""

_SKYMODEL_WITHOUT_PATCHES = """\
# (Name, Type, Ra, Dec, I) = format

SRC_A, POINT, 10deg, -20deg, 5.0
SRC_B, POINT, 11deg, -21deg, 1.0
"""


@pytest.fixture
def skymodel(tmp_path):
    """A sky model with two patches, written to a file."""
    path = tmp_path / "sky.skymodel"
    path.write_text(_SKYMODEL_WITH_PATCHES)
    return os.fspath(path)


@pytest.fixture
def model_vis():
    """Model visibilities matching the ``vis_data`` fixture."""
    shape = (_N_TIME, _N_BL, _N_FREQ, 4)
    rng = np.random.default_rng(7)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )


@pytest.fixture
def vis_data():
    """Observed visibilities with flags and weights."""
    shape = (_N_TIME, _N_BL, _N_FREQ, 4)
    rng = np.random.default_rng(42)
    vis = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    flags = np.zeros(shape, dtype=bool)
    flags[0, 0, 0, :] = True
    weight = rng.uniform(size=shape).astype(np.float32)
    return vis, flags, weight


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
        antenna_names=["ANT0", "ANT1", "ANT2"],
        antenna_positions=np.zeros((_N_ANT, 3)),
        antenna_diameters=np.full(_N_ANT, 38.0),
        frequency=np.array([100e6, 101e6]),
        channel_bandwidth=np.array([1e6, 1e6]),
        time=np.arange(_N_TIME) * 10.0 + 5e9,
        integration_time=10.0,
        uvw=np.arange(_N_TIME * _N_BL * 3, dtype=float).reshape(
            _N_TIME, _N_BL, 3
        ),
        # At patch p_other of the sky model fixture
        phasecentre=(np.radians(11.0), np.radians(-21.0)),
    )


@pytest.fixture
def mock_dp3(monkeypatch, gain_data):
    """
    Replace the ``dp3`` package and the helpers of ``dp3_ddecal_solve``
    with mocks, all children of a single ``manager`` mock.
    """
    manager = MagicMock(name="manager")

    dp3 = manager.dp3
    parameterset = MagicMock(name="dp3.parameterset")
    dp3.parameterset = parameterset
    monkeypatch.setitem(sys.modules, "dp3", dp3)
    monkeypatch.setitem(sys.modules, "dp3.parameterset", parameterset)

    parsets = [MagicMock(name="parset"), MagicMock(name="null_parset")]
    parameterset.ParameterSet.side_effect = parsets

    ddecal_step = manager.ddecal_step
    dp3.make_step.side_effect = [ddecal_step, MagicMock(name="null_step")]

    buffers = []

    def _make_buffer():
        buffers.append(MagicMock(name=f"buffer{len(buffers)}"))
        return buffers[-1]

    dp3.DPBuffer.side_effect = _make_buffer

    # Only the diagonal of the first antenna is solved
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
        f"{_MODULE}.set_dp3_n_threads", manager.set_dp3_n_threads
    ), patch(
        f"{_MODULE}.read_h5parm_gains", manager.read_h5parm_gains
    ), patch(
        f"{_MODULE}._create_dp_info", manager.create_dp_info
    ), patch(
        f"{_MODULE}._register_model_direction",
        manager.register_model_direction,
    ):
        yield {
            "manager": manager,
            "dp3": dp3,
            "parset": parsets[0],
            "ddecal_step": ddecal_step,
            "buffers": buffers,
            "dp3_gain": dp3_gain,
            "dp3_weight": dp3_weight,
            "solved": solved,
        }


def _solve(vis_data, gain_data, observation, skymodel, **kwargs):
    """Call ``dp3_ddecal_solve`` with the fixture data."""
    vis, flags, weight = vis_data
    gain, gain_weight, residual = gain_data
    model_flags = kwargs.pop("model_flags", None)
    model_vis = kwargs.pop("model_vis", None)
    return dp3_ddecal_solve(
        vis,
        flags,
        weight,
        model_vis,
        model_flags,
        gain,
        gain_weight,
        residual,
        _ANT1,
        _ANT2,
        skymodel_path=skymodel,
        observation=observation,
        **kwargs,
    )


def _parset_values(parset):
    return {
        add_call.args[0]: add_call.args[1]
        for add_call in parset.add.call_args_list
    }


@patch(f"{_MODULE}.require_dp3")
def test_dp3_ddecal_solver_should_be_registered(require_dp3_mock):
    solver = Solver.get_solver(
        "dp3_ddecal",
        skymodel_path="sky.skymodel",
        direction=["p_main"],
        crosspol=True,
        niter=10,
    )

    assert isinstance(solver, Dp3DdecalSolver)
    assert solver.requires_observation is True
    assert solver.direction == ["p_main"]
    assert solver.mode == "fulljones"
    assert solver.niter == 10
    require_dp3_mock.assert_called_once_with("dp3_ddecal solver")


@patch(f"{_MODULE}.require_dp3")
def test_dp3_ddecal_solver_should_not_require_skymodel(_require_dp3_mock):
    solver = Dp3DdecalSolver()

    assert solver.skymodel_path is None
    assert solver.direction is None
    assert solver.mode == "diagonal"


@patch(f"{_MODULE}.dp3_ddecal_solve")
@patch(f"{_MODULE}.require_dp3")
def test_dp3_ddecal_solver_solve_should_call_dp3_ddecal_solve(
    _require_dp3_mock, dp3_ddecal_solve_mock
):
    solver = Dp3DdecalSolver(
        skymodel_path="sky.skymodel",
        direction=["p_main"],
        phase_only=True,
        niter=10,
        tol=1e-3,
        n_threads=2,
    )
    arrays = [MagicMock(name=f"array{idx}") for idx in range(10)]
    observation = MagicMock(name="observation")

    result = solver.solve(*arrays, observation=observation)

    # With a sky model, DDECal predicts the model, so model_vis is not used
    expected_arrays = arrays[:3] + [None] + arrays[4:]
    dp3_ddecal_solve_mock.assert_called_once_with(
        *expected_arrays,
        skymodel_path="sky.skymodel",
        observation=observation,
        direction=["p_main"],
        mode="diagonalphase",
        niter=10,
        tol=1e-3,
        n_threads=2,
    )
    assert result == dp3_ddecal_solve_mock.return_value


@patch(f"{_MODULE}.dp3_ddecal_solve")
@patch(f"{_MODULE}.require_dp3")
def test_dp3_ddecal_solver_solve_should_pass_model_vis_without_skymodel(
    _require_dp3_mock, dp3_ddecal_solve_mock
):
    solver = Dp3DdecalSolver(crosspol=True)
    arrays = [MagicMock(name=f"array{idx}") for idx in range(10)]

    solver.solve(*arrays)

    dp3_ddecal_solve_mock.assert_called_once_with(
        *arrays,
        skymodel_path=None,
        observation=None,
        direction=None,
        mode="fulljones",
        niter=50,
        tol=1e-6,
        n_threads=None,
    )


def test_dp3_ddecal_solve_should_solve_for_phase_centre_patch_by_default(
    mock_dp3, vis_data, gain_data, observation, skymodel
):
    _solve(
        vis_data,
        gain_data,
        observation,
        skymodel,
        mode="fulljones",
        niter=10,
        tol=1e-3,
    )

    mock_dp3["manager"].require_dp3.assert_called_once_with("dp3_ddecal_solve")
    assert _parset_values(mock_dp3["parset"]) == {
        "ddecal.sourcedb": skymodel,
        "ddecal.directions": "[[p_other]]",
        "ddecal.h5parm": ANY,
        "ddecal.mode": "fulljones",
        "ddecal.solint": "2",
        "ddecal.nchan": "2",
        "ddecal.maxiter": "10",
        "ddecal.tolerance": "0.001",
    }
    mock_dp3["dp3"].make_step.assert_has_calls(
        [
            call(
                "ddecal",
                mock_dp3["parset"],
                "ddecal.",
                mock_dp3["dp3"].MsType.regular,
            )
        ]
    )
    mock_dp3["manager"].create_dp_info.assert_called_once_with(
        _ANT1, _ANT2, _N_ANT, _N_TIME, _N_FREQ, observation
    )
    mock_dp3["manager"].register_model_direction.assert_not_called()
    mock_dp3["ddecal_step"].set_info.assert_called_once_with(
        mock_dp3["manager"].create_dp_info.return_value
    )
    for buffer in mock_dp3["buffers"]:
        buffer.add_data.assert_not_called()


def test_dp3_ddecal_solve_should_use_given_direction(
    mock_dp3, vis_data, gain_data, observation, skymodel
):
    _solve(vis_data, gain_data, observation, skymodel, direction=["p_main"])

    assert (
        _parset_values(mock_dp3["parset"])["ddecal.directions"] == "[[p_main]]"
    )


@pytest.mark.parametrize("direction", [[], [""], ["p_main", 3]])
def test_dp3_ddecal_solve_should_raise_for_invalid_direction(
    mock_dp3, vis_data, gain_data, observation, skymodel, direction
):
    with pytest.raises(ValueError, match="non-empty list of patch names"):
        _solve(vis_data, gain_data, observation, skymodel, direction=direction)

    mock_dp3["dp3"].make_step.assert_not_called()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"skymodel": ""}, "skymodel_path and observation are required"),
        ({"observation": None}, "skymodel_path and observation are required"),
        ({"mode": "tec"}, "unsupported mode 'tec'"),
    ],
)
def test_dp3_ddecal_solve_should_validate_arguments(
    mock_dp3, vis_data, gain_data, observation, skymodel, kwargs, match
):
    kwargs = {"observation": observation, "skymodel": skymodel, **kwargs}

    with pytest.raises(ValueError, match=match):
        _solve(vis_data, gain_data, **kwargs)

    mock_dp3["dp3"].make_step.assert_not_called()


def test_dp3_ddecal_solve_should_reuse_given_model(
    mock_dp3, vis_data, gain_data, observation, model_vis
):
    manager = mock_dp3["manager"]

    _solve(
        vis_data,
        gain_data,
        observation,
        None,
        model_vis=model_vis,
        mode="fulljones",
    )

    assert _parset_values(mock_dp3["parset"]) == {
        "ddecal.reusemodel": "[modelinfo.modeldata]",
        "ddecal.h5parm": ANY,
        "ddecal.mode": "fulljones",
        "ddecal.solint": "2",
        "ddecal.nchan": "2",
        "ddecal.maxiter": "50",
        "ddecal.tolerance": "1e-06",
    }
    # The model direction is registered in the DPInfo given to DDECal
    manager.register_model_direction.assert_called_once_with(
        manager.create_dp_info.return_value, observation.phasecentre, ANY
    )
    mock_dp3["ddecal_step"].set_info.assert_called_once_with(
        manager.register_model_direction.return_value
    )
    for time_idx, buffer in enumerate(mock_dp3["buffers"]):
        buffer.set_time.assert_called_once_with(observation.time[time_idx])
        buffer.add_data.assert_called_once_with("modelinfo.modeldata")
        name, extra_data = buffer.set_extra_data.call_args.args
        assert name == "modelinfo.modeldata"
        np.testing.assert_array_equal(
            extra_data,
            expand_polarizations(model_vis[time_idx], np.complex64),
        )


def test_dp3_ddecal_solve_should_register_model_while_holding_lock(
    mock_dp3, vis_data, gain_data, model_vis
):
    _solve(vis_data, gain_data, None, None, model_vis=model_vis)

    call_names = [name for name, _, _ in mock_dp3["manager"].mock_calls]
    register_idx = call_names.index("register_model_direction")
    assert call_names.index("get_dp3_lock().__enter__") < register_idx
    assert register_idx < call_names.index("get_dp3_lock().__exit__")


def test_dp3_ddecal_solve_should_use_placeholder_metadata_with_model(
    mock_dp3, vis_data, gain_data, model_vis
):
    _solve(vis_data, gain_data, None, None, model_vis=model_vis)

    manager = mock_dp3["manager"]
    manager.create_dp_info.assert_called_once_with(
        _ANT1, _ANT2, _N_ANT, _N_TIME, _N_FREQ, None
    )
    manager.register_model_direction.assert_called_once_with(
        manager.create_dp_info.return_value, (0.0, 0.0), ANY
    )
    for time_idx, buffer in enumerate(mock_dp3["buffers"]):
        buffer.set_time.assert_called_once_with(time_idx + 0.5)
        np.testing.assert_array_equal(
            buffer.set_uvw.call_args.args[0], np.zeros((_N_BL, 3))
        )


def test_dp3_ddecal_solve_should_raise_for_mismatched_model_shape(
    mock_dp3, vis_data, gain_data, model_vis
):
    with pytest.raises(ValueError, match="model_vis must match"):
        _solve(vis_data, gain_data, None, None, model_vis=model_vis[:-1])

    mock_dp3["dp3"].make_step.assert_not_called()


def test_register_model_direction_should_return_info_of_keepmodel_ddecal(
    monkeypatch, tmp_path
):
    dp3 = MagicMock(name="dp3")
    parameterset = MagicMock(name="dp3.parameterset")
    monkeypatch.setitem(sys.modules, "dp3", dp3)
    monkeypatch.setitem(sys.modules, "dp3.parameterset", parameterset)
    parset = MagicMock(name="parset")
    parameterset.ParameterSet.side_effect = [parset, MagicMock()]
    step = MagicMock(name="step")
    dp3.make_step.side_effect = [step, MagicMock(name="null_step")]
    dpinfo = MagicMock(name="dpinfo")

    info = _register_model_direction(
        dpinfo, (np.float64(0.25), np.float64(-0.5)), os.fspath(tmp_path)
    )

    skymodel_path = os.fspath(tmp_path / "model_direction.skymodel")
    assert _parset_values(parset) == {
        "modelinfo.sourcedb": skymodel_path,
        "modelinfo.onlypredict": "true",
        "modelinfo.keepmodel": "true",
    }
    with open(skymodel_path, encoding="utf-8") as skymodel:
        assert skymodel.read() == (
            "FORMAT = Name, Type, Ra, Dec, I\n"
            "modeldata, POINT, 0.25rad, -0.5rad, 1.0\n"
        )
    dp3.make_step.assert_any_call(
        "ddecal", parset, "modelinfo.", dp3.MsType.regular
    )
    step.set_info.assert_called_once_with(dpinfo)
    # Only the DPInfo is used, no data is processed
    step.process.assert_not_called()
    assert info is step.info_out


def test_dp3_ddecal_solve_should_raise_for_mismatched_observation(
    mock_dp3, vis_data, gain_data, observation, skymodel
):
    observation.time = observation.time[:-1]

    with pytest.raises(ValueError, match="observation time axis"):
        _solve(vis_data, gain_data, observation, skymodel)


def test_dp3_ddecal_solve_should_stream_timeslots(
    mock_dp3, vis_data, gain_data, observation, skymodel
):
    vis, flags, weight = vis_data
    model_flags = np.zeros(flags.shape, dtype=bool)
    model_flags[1, 2, 1, :] = True

    _solve(vis_data, gain_data, observation, skymodel, model_flags=model_flags)

    buffers = mock_dp3["buffers"]
    assert len(buffers) == _N_TIME
    mock_dp3["ddecal_step"].process.assert_has_calls(
        [call(buffer) for buffer in buffers]
    )
    for time_idx, buffer in enumerate(buffers):
        buffer.set_time.assert_called_once_with(observation.time[time_idx])
        # DP3 uses the opposite uvw sign convention
        np.testing.assert_array_equal(
            buffer.set_uvw.call_args.args[0], -observation.uvw[time_idx]
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
            expand_polarizations(
                flags[time_idx] | model_flags[time_idx], bool
            ),
        )
    mock_dp3["ddecal_step"].finish.assert_called_once_with()


def test_dp3_ddecal_solve_should_read_solutions_after_releasing_lock(
    mock_dp3, vis_data, gain_data, observation, skymodel
):
    _solve(vis_data, gain_data, observation, skymodel)

    h5parm_path = _parset_values(mock_dp3["parset"])["ddecal.h5parm"]
    assert os.path.basename(h5parm_path) == "ddecal.h5"
    mock_dp3["manager"].read_h5parm_gains.assert_called_once_with(h5parm_path)
    # The temporary directory is removed after reading the solutions
    assert not os.path.exists(os.path.dirname(h5parm_path))

    call_names = [name for name, _, _ in mock_dp3["manager"].mock_calls]
    exit_idx = call_names.index("get_dp3_lock().__exit__")
    assert call_names.index("ddecal_step.finish") < exit_idx
    assert exit_idx < call_names.index("read_h5parm_gains")


def test_dp3_ddecal_solve_should_return_gain_weight_and_residual(
    mock_dp3, vis_data, gain_data, observation, skymodel
):
    gain, gain_weight, residual = gain_data
    solved = mock_dp3["solved"]

    new_gain, new_weight, new_residual = _solve(
        vis_data, gain_data, observation, skymodel
    )

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


def test_split_skymodel_line_should_keep_lists_and_empty_fields():
    assert _split_skymodel_line("A, POINT,, [-0.7, 0.1], 5") == [
        "A",
        "POINT",
        "",
        "[-0.7, 0.1]",
        "5",
    ]


def _assert_positions(patches, expected_deg):
    """Compare patch positions in radians with expected ones in degrees."""
    assert list(patches) == list(expected_deg)
    for name, (ra, dec) in expected_deg.items():
        np.testing.assert_allclose(
            np.degrees(patches[name]), (ra, dec), atol=1e-6
        )


def test_read_skymodel_patches_should_return_flux_weighted_directions(
    skymodel,
):
    patches = _read_skymodel_patches(skymodel)

    # p_main: SRC_A (5 Jy at ra 10deg) and SRC_B (1 Jy at ra 10.1deg)
    expected_ra = np.degrees(
        np.arctan2(
            5 * np.sin(np.radians(10.0)) + np.sin(np.radians(10.1)),
            5 * np.cos(np.radians(10.0)) + np.cos(np.radians(10.1)),
        )
    )
    np.testing.assert_allclose(
        np.degrees(patches["p_main"]), (expected_ra, -20.0), atol=1e-4
    )
    np.testing.assert_allclose(
        np.degrees(patches["p_other"]), (11.0, -21.0), atol=1e-6
    )
    assert list(patches) == ["p_main", "p_other"]


def test_read_skymodel_patches_should_use_source_names_without_patches(
    tmp_path,
):
    path = tmp_path / "sky.txt"
    path.write_text(_SKYMODEL_WITHOUT_PATCHES)

    _assert_positions(
        _read_skymodel_patches(os.fspath(path)),
        {"SRC_A": (10.0, -20.0), "SRC_B": (11.0, -21.0)},
    )


def test_read_skymodel_patches_should_use_patch_position_without_flux(
    tmp_path,
):
    path = tmp_path / "sky.txt"
    path.write_text(
        "FORMAT = Name, Type, Patch, Ra, Dec, I\n"
        ",,p_empty,12deg,-22deg\n"
        ",,p_unplaced,,\n"
        "SRC_A,POINT,p_empty,13deg,-23deg,0.0\n"
    )

    patches = _read_skymodel_patches(os.fspath(path))

    _assert_positions(
        {"p_empty": patches["p_empty"]}, {"p_empty": (12.0, -22.0)}
    )
    assert patches["p_unplaced"] is None


def test_read_skymodel_patches_should_raise_for_sourcedb(tmp_path):
    with pytest.raises(ValueError, match="direction must be given"):
        _read_skymodel_patches(os.fspath(tmp_path))


def test_read_skymodel_patches_should_raise_without_format(tmp_path):
    path = tmp_path / "sky.txt"
    path.write_text("SRC_A, POINT, 10deg, -20deg, 5.0\n")

    with pytest.raises(ValueError, match="no format line"):
        _read_skymodel_patches(os.fspath(path))


@pytest.mark.parametrize(
    "value,expected_deg",
    [
        ("197.5deg", 197.5),
        ("13:10:00", 197.5),
        ("-22.30.00", -22.5),
        (f"{np.radians(30.0)}rad", 30.0),
    ],
)
def test_parse_skymodel_angle(value, expected_deg):
    assert np.degrees(_parse_skymodel_angle(value)) == pytest.approx(
        expected_deg
    )


def test_parse_skymodel_angle_should_raise_without_unit():
    with pytest.raises(ValueError, match="has no angle unit"):
        _parse_skymodel_angle("1.2")


@pytest.mark.parametrize(
    "phasecentre_deg,expected",
    [((10.0, -20.0), "p_main"), ((11.1, -21.1), "p_other")],
)
def test_find_phase_centre_patch_should_return_nearest_patch(
    skymodel, phasecentre_deg, expected
):
    phasecentre = tuple(np.radians(phasecentre_deg))

    assert _find_phase_centre_patch(skymodel, phasecentre) == expected


def test_find_phase_centre_patch_should_raise_without_positions(tmp_path):
    path = tmp_path / "sky.txt"
    path.write_text("FORMAT = Name, Type, Patch, Ra, Dec, I\n,,p1,,\n")

    with pytest.raises(ValueError, match="no patch with a position"):
        _find_phase_centre_patch(os.fspath(path), (0.0, 0.0))
