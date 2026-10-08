import h5py
import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.data_managers.h5parm import (
    create_clock_soltab_datasets,
    create_soltab_datasets,
    create_soltab_group,
    read_h5parm_gains,
    read_soltab,
    to_null_terminated_bytes,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable

TIME = np.array([1.0, 2.0])
ANT = to_null_terminated_bytes(["ANT0", "ANT1", "ANT2"])
ANT_NAMES = ["ANT0", "ANT1", "ANT2"]
FREQ = np.array([100e6, 110e6, 120e6, 130e6])
LINEAR_POLS = np.array([b"XX", b"XY", b"YX", b"YY"])


@pytest.fixture
def h5file(tmp_path):
    with h5py.File(tmp_path / "test.h5parm", "w") as h5f:
        yield h5f


def test_should_create_soltab_group(h5file):
    solset = h5file.create_group("sol000")

    soltab = create_soltab_group(solset, "phase")

    assert soltab == h5file["sol000/phase000"]
    assert soltab.attrs["TITLE"] == b"phase"


def test_should_create_soltab_datasets(h5file):
    # Gaintable in h5parm layout, as passed by export_gaintable_to_h5parm
    freq = np.array([100e6, 110e6])
    pol = to_null_terminated_bytes(["XX", "XY", "YX", "YY"])
    gaintable = xr.Dataset(
        {"gain": (("time", "ant", "freq", "pol"), np.ones((2, 3, 2, 4)))},
        coords={"time": TIME, "ant": ANT, "freq": freq, "pol": pol},
    )
    soltab = h5file.create_group("amplitude000")

    val, weight = create_soltab_datasets(soltab, gaintable)

    np.testing.assert_array_equal(soltab["time"][...], TIME)
    np.testing.assert_array_equal(soltab["ant"][...], ANT)
    np.testing.assert_array_equal(soltab["freq"][...], freq)
    np.testing.assert_array_equal(soltab["pol"][...], pol)

    assert val == soltab["val"]
    assert weight == soltab["weight"]
    for dataset in (val, weight):
        assert dataset.shape == (2, 3, 2, 4)
        assert dataset.dtype == float
        assert dataset.attrs["AXES"] == b"time,ant,freq,pol"


def test_should_create_clock_soltab_datasets(h5file):
    # Delaytable in h5parm layout, as passed by export_clock_to_h5parm
    pol = to_null_terminated_bytes(["XX", "YY"])
    delaytable = DelayTable.constructor(
        delay=np.zeros((2, 3, 2)),
        offset=np.zeros((2, 3, 2)),
        time=TIME,
        antenna=ANT,
        pol=pol,
    ).rename({"antenna": "ant"})
    soltab = h5file.create_group("clock000")

    val, offset = create_clock_soltab_datasets(soltab, delaytable)

    np.testing.assert_array_equal(soltab["time"][...], TIME)
    np.testing.assert_array_equal(soltab["ant"][...], ANT)
    np.testing.assert_array_equal(soltab["pol"][...], pol)

    assert val == soltab["val"]
    assert offset == soltab["offset"]
    for dataset in (val, offset):
        assert dataset.shape == (2, 3, 2)
        assert dataset.dtype == float
        assert dataset.attrs["AXES"] == b"time,ant,pol"


def _write_soltab(
    solset: h5py.Group,
    name: str,
    val: np.ndarray,
    axes: dict[str, np.ndarray],
    weight: np.ndarray | None = None,
) -> h5py.Group:
    """Write a soltab with the given axes (in order) and coordinates."""
    soltab = solset.create_group(name)
    for axis, coords in axes.items():
        soltab.create_dataset(axis, data=coords)

    datasets = (
        {"val": val} if weight is None else {"val": val, "weight": weight}
    )
    for dataset_name, data in datasets.items():
        dataset = soltab.create_dataset(dataset_name, data=data)
        dataset.attrs["AXES"] = np.bytes_(",".join(axes))

    return soltab


def _random(*shape: int) -> np.ndarray:
    return np.random.default_rng(42).uniform(0.1, 1.0, size=shape)


def test_should_read_inst_soltab(h5file):
    val, weight = _random(2, 3, 4, 4), _random(2, 3, 4, 4)
    axes = {"time": TIME, "ant": ANT, "freq": FREQ, "pol": LINEAR_POLS}
    soltab = _write_soltab(h5file, "amplitude000", val, axes, weight)

    data = read_soltab(soltab)

    np.testing.assert_array_equal(data.values, val.reshape(2, 3, 4, 2, 2))
    np.testing.assert_array_equal(data.weights, weight.reshape(2, 3, 4, 2, 2))
    assert data.solved.shape == (2, 3, 4, 2, 2)
    assert np.all(data.solved)
    np.testing.assert_array_equal(data.time, TIME)
    np.testing.assert_array_equal(data.frequency, FREQ)
    assert data.antenna == ANT_NAMES


def test_should_read_dp3_soltab(h5file):
    # DP3 gaincal writes the frequency axis before the antenna axis
    val, weight = _random(2, 4, 3, 4), _random(2, 4, 3, 4)
    axes = {"time": TIME, "freq": FREQ, "ant": ANT, "pol": LINEAR_POLS}
    soltab = _write_soltab(h5file, "amplitude000", val, axes, weight)

    data = read_soltab(soltab)

    expected_val = val.transpose(0, 2, 1, 3).reshape(2, 3, 4, 2, 2)
    expected_weight = weight.transpose(0, 2, 1, 3).reshape(2, 3, 4, 2, 2)
    np.testing.assert_array_equal(data.values, expected_val)
    np.testing.assert_array_equal(data.weights, expected_weight)
    np.testing.assert_array_equal(data.time, TIME)
    np.testing.assert_array_equal(data.frequency, FREQ)
    assert data.antenna == ANT_NAMES


def test_should_read_axes_attribute_stored_as_str(h5file):
    axes = {"time": TIME, "ant": ANT, "freq": FREQ}
    soltab = _write_soltab(h5file, "phase000", _random(2, 3, 4), axes)
    soltab["val"].attrs["AXES"] = "time,ant,freq"

    data = read_soltab(soltab)

    assert data.values.shape == (2, 3, 4, 2, 2)


@pytest.mark.parametrize("pols", [["XX", "YY"], ["RR", "LL"]])
def test_should_fill_cross_pols_with_zeros_for_diagonal_soltab(h5file, pols):
    val, weight = _random(2, 3, 4, 2), _random(2, 3, 4, 2)
    axes = {"time": TIME, "ant": ANT, "freq": FREQ, "pol": np.array(pols, "S")}
    soltab = _write_soltab(h5file, "phase000", val, axes, weight)

    data = read_soltab(soltab)

    np.testing.assert_array_equal(data.values[..., 0, 0], val[..., 0])
    np.testing.assert_array_equal(data.values[..., 1, 1], val[..., 1])
    np.testing.assert_array_equal(data.weights[..., 0, 0], weight[..., 0])
    np.testing.assert_array_equal(data.weights[..., 1, 1], weight[..., 1])
    for r1, r2 in [(0, 1), (1, 0)]:
        assert np.all(data.values[..., r1, r2] == 0)
        assert np.all(data.weights[..., r1, r2] == 0)
    np.testing.assert_array_equal(data.solved[0, 0, 0], np.eye(2, dtype=bool))


def test_should_use_scalar_soltab_for_both_diagonal_terms(h5file):
    val = _random(2, 3, 4)
    axes = {"time": TIME, "ant": ANT, "freq": FREQ}
    soltab = _write_soltab(h5file, "phase000", val, axes, np.ones_like(val))

    data = read_soltab(soltab)

    np.testing.assert_array_equal(data.values[..., 0, 0], val)
    np.testing.assert_array_equal(data.values[..., 1, 1], val)
    assert np.all(data.values[..., 0, 1] == 0)
    np.testing.assert_array_equal(data.solved[0, 0, 0], np.eye(2, dtype=bool))


def test_should_restore_squeezed_axes(h5file):
    val = _random(3, 4)
    axes = {"ant": ANT, "pol": LINEAR_POLS}
    soltab = _write_soltab(h5file, "phase000", val, axes, np.ones_like(val))

    data = read_soltab(soltab)

    np.testing.assert_array_equal(data.values, val.reshape(1, 3, 1, 2, 2))
    assert data.time is None
    assert data.frequency is None
    assert data.antenna == ANT_NAMES


def test_should_set_antenna_to_none_when_antenna_axis_absent(h5file):
    axes = {"time": TIME, "freq": FREQ}
    soltab = _write_soltab(h5file, "phase000", _random(2, 4), axes)

    data = read_soltab(soltab)

    assert data.values.shape == (2, 1, 4, 2, 2)
    assert data.antenna is None


def test_should_use_unit_weights_when_weight_absent(h5file):
    axes = {"time": TIME, "ant": ANT, "freq": FREQ, "pol": LINEAR_POLS}
    soltab = _write_soltab(h5file, "phase000", _random(2, 3, 4, 4), axes)

    data = read_soltab(soltab)

    assert np.all(data.weights == 1)


def test_should_raise_exception_for_unsupported_soltab_axes(h5file):
    axes = {"time": TIME, "ant": ANT, "dir": np.array([b"POINTING"])}
    soltab = _write_soltab(h5file, "phase000", _random(2, 3, 1), axes)

    with pytest.raises(ValueError, match=r"unsupported axes \['dir'\]"):
        read_soltab(soltab)


def test_should_raise_exception_for_unsupported_pol(h5file):
    axes = {"ant": ANT, "pol": np.array([b"XX", b"I"])}
    soltab = _write_soltab(h5file, "phase000", _random(3, 2), axes)

    with pytest.raises(ValueError, match="Unsupported polarisation 'I'"):
        read_soltab(soltab)


def _write_h5parm(
    path: str,
    amplitude: np.ndarray | None = None,
    phase: np.ndarray | None = None,
    solset: str = "sol000",
) -> None:
    """Write an h5parm in DP3 layout with the given soltab values."""
    axes = {"time": TIME, "freq": FREQ, "ant": ANT, "pol": LINEAR_POLS}
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group(solset)
        for name, val in (("amplitude000", amplitude), ("phase000", phase)):
            if val is not None:
                _write_soltab(group, name, val, axes, np.ones_like(val))


def test_should_read_h5parm_gains(tmp_path):
    path = str(tmp_path / "gains.h5parm")
    amplitude, phase = _random(2, 4, 3, 4), _random(2, 4, 3, 4)
    with h5py.File(path, "w") as h5f:
        axes = {"time": TIME, "freq": FREQ, "ant": ANT, "pol": LINEAR_POLS}
        solset = h5f.create_group("sol000")
        weights = np.full_like(amplitude, 0.5), np.full_like(phase, 0.25)
        _write_soltab(solset, "amplitude000", amplitude, axes, weights[0])
        _write_soltab(solset, "phase000", phase, axes, weights[1])

    gain, weight, solved, soltab = read_h5parm_gains(path)

    def to_jones(data):
        return data.transpose(0, 2, 1, 3).reshape(2, 3, 4, 2, 2)

    np.testing.assert_allclose(
        gain, to_jones(amplitude) * np.exp(1j * to_jones(phase))
    )
    # minimum of the amplitude and phase weights
    assert np.all(weight == 0.25)
    assert np.all(solved)
    np.testing.assert_array_equal(soltab.time, TIME)
    np.testing.assert_array_equal(soltab.frequency, FREQ)
    assert soltab.antenna == ANT_NAMES


def test_should_read_h5parm_gains_from_given_solset(tmp_path):
    path = str(tmp_path / "gains.h5parm")
    _write_h5parm(path, amplitude=_random(2, 4, 3, 4), solset="sol001")

    gain, *_ = read_h5parm_gains(path, solset="sol001")

    assert gain.shape == (2, 3, 4, 2, 2)


def test_should_read_h5parm_gains_with_amplitude_only(tmp_path):
    path = str(tmp_path / "gains.h5parm")
    amplitude = _random(2, 4, 3, 4)
    _write_h5parm(path, amplitude=amplitude)

    gain, _, _, soltab = read_h5parm_gains(path)

    np.testing.assert_allclose(gain.imag, 0)
    np.testing.assert_array_equal(soltab.time, TIME)


def test_should_read_h5parm_gains_with_phase_only(tmp_path):
    path = str(tmp_path / "gains.h5parm")
    _write_h5parm(path, phase=_random(2, 4, 3, 4))

    gain, _, _, soltab = read_h5parm_gains(path)

    np.testing.assert_allclose(np.abs(gain[..., 0, 0]), 1)
    np.testing.assert_array_equal(soltab.time, TIME)


def test_should_raise_exception_when_h5parm_has_no_gain_soltabs(tmp_path):
    path = str(tmp_path / "gains.h5parm")
    _write_h5parm(path)

    with pytest.raises(ValueError, match="No amplitude000 or phase000"):
        read_h5parm_gains(path)
