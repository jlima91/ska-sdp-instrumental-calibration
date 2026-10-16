import numpy as np
import pytest
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    _check_antennas,
    create_gaintable_from_h5parm,
    create_gaintable_from_visibility,
    divide_bandpass_by_ref_ant_preserve_phase,
    reset_gaintable,
)
from ska_sdp_instrumental_calibration.data_managers.h5parm import SoltabData


def test_should_raise_exception_for_jonestype(generate_vis):
    vis, jones = generate_vis
    with pytest.raises(ValueError, match="Unknown Jones type X"):
        create_gaintable_from_visibility(vis, jones_type="X")


def test_should_generate_gaintable_with_defaults(generate_vis):
    vis, jones = generate_vis
    gaintable = create_gaintable_from_visibility(vis)

    expected_frequency = np.mean(vis.frequency.data, keepdims=True)
    gain_shape = (len(vis.time.data), vis.visibility_acc.nants, 1, 4)
    residual_shape = (len(vis.time.data), 1, 4)

    np.testing.assert_allclose(gaintable.frequency.data, expected_frequency)
    assert gaintable.CALPARAM_GAIN.shape == gain_shape
    assert gaintable.CALPARAM_WEIGHT.shape == gain_shape
    assert gaintable.CALPARAM_RESIDUAL.shape == residual_shape
    assert gaintable.attrs["soln_interval_slices"] == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
    ]


@pytest.mark.parametrize(
    "polarisation",
    [["XX", "XY", "YX", "YY"], ["RR", "RL", "LR", "LL"]],
    ids=["linear", "circular"],
)
def test_should_take_names_and_pols_from_visibility(
    generate_vis, polarisation
):
    vis, _ = generate_vis
    vis = vis.assign_coords(polarisation=polarisation)

    gaintable = create_gaintable_from_visibility(vis)

    assert gaintable.polarization.data.tolist() == polarisation
    assert (
        gaintable.antenna_name.data.tolist()
        == vis.configuration.names.data.tolist()
    )


@pytest.mark.parametrize(
    "polarisation",
    [
        ["XX", "YY", "XY", "YX"],
        ["I", "Q", "U", "V"],
        ["XX", "YY"],
        ["I"],
    ],
    ids=["linearFITS", "stokesIQUV", "linearnp", "stokesI"],
)
def test_should_raise_for_unsupported_visibility_pols(
    generate_vis, polarisation
):
    vis, _ = generate_vis
    vis = vis.isel(polarisation=slice(0, len(polarisation)))
    vis = vis.assign_coords(polarisation=polarisation)

    with pytest.raises(
        ValueError, match="Unsupported visibility polarizations"
    ):
        create_gaintable_from_visibility(vis)


def test_should_generate_gaintable_for_jonetypes_B(generate_vis):
    vis, jones = generate_vis
    vis = vis.chunk(frequency=1)
    gaintable = create_gaintable_from_visibility(vis, jones_type="B")

    gain_shape = (
        len(vis.time.data),
        vis.visibility_acc.nants,
        len(vis.frequency.data),
        4,
    )

    np.testing.assert_allclose(gaintable.frequency.data, vis.frequency.data)
    assert gaintable.CALPARAM_GAIN.shape == gain_shape
    assert gaintable.CALPARAM_WEIGHT.shape == gain_shape
    assert gaintable.attrs["soln_interval_slices"] == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
    ]


def test_should_skip_default_chunk(generate_vis):
    vis, jones = generate_vis
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", skip_default_chunk=True
    )

    gain_shape = (
        len(vis.time.data),
        vis.visibility_acc.nants,
        len(vis.frequency.data),
        4,
    )

    np.testing.assert_allclose(gaintable.frequency.data, vis.frequency.data)
    assert gaintable.CALPARAM_GAIN.shape == gain_shape
    assert gaintable.CALPARAM_WEIGHT.shape == gain_shape
    assert gaintable.attrs["soln_interval_slices"] == [
        slice(0, 1, 1),
        slice(1, 2, 1),
        slice(2, 3, 1),
    ]


@patch("ska_sdp_instrumental_calibration.data_managers.gaintable.jones_to_pol")
@patch("ska_sdp_instrumental_calibration.data_managers.gaintable.da")
def test_should_reset_gaintable(da_mock, jones_to_pol_mock):
    gaintable = MagicMock(name="gaintable")
    gaintable.calibration_set.nrec = 2
    r_gaintable = reset_gaintable(gaintable)
    da_mock.eye.assert_called_once_with(
        gaintable.calibration_set.nrec,
        dtype=gaintable.CALPARAM_GAIN.dtype,
    )
    jones_to_pol_mock.assert_called_once_with(da_mock.eye.return_value)

    da_mock.broadcast_to.assert_called_once_with(
        jones_to_pol_mock.return_value,
        gaintable.CALPARAM_GAIN.shape,
    )

    da_mock.ones.assert_called_once_with(
        gaintable.CALPARAM_WEIGHT.shape, dtype=gaintable.CALPARAM_WEIGHT.dtype
    )

    da_mock.zeros.assert_called_once_with(
        gaintable.CALPARAM_RESIDUAL.shape,
        dtype=gaintable.CALPARAM_RESIDUAL.dtype,
    )

    gaintable.copy.assert_called_once_with(deep=True)
    assert r_gaintable == gaintable.copy.return_value
    assert r_gaintable.CALPARAM_GAIN.data == da_mock.broadcast_to.return_value
    assert r_gaintable.CALPARAM_WEIGHT.data == da_mock.ones.return_value
    assert r_gaintable.CALPARAM_RESIDUAL.data == da_mock.zeros.return_value


def test_should_divide_bandpass_by_ref_ant_and_preserve_phase(generate_vis):
    vis, _ = generate_vis
    vis = vis.chunk(frequency=1)
    gaintable = create_gaintable_from_visibility(vis, jones_type="B")

    actual_gaintable = divide_bandpass_by_ref_ant_preserve_phase(gaintable, 0)
    complex_gains = actual_gaintable.CALPARAM_GAIN.data

    x_angle = np.angle(complex_gains[:, 0, :, 0])
    y_angle = np.angle(complex_gains[:, 0, :, 3])

    actual_amp = np.abs(complex_gains)
    expected_amp = np.abs(gaintable.CALPARAM_GAIN.data)

    assert np.allclose(x_angle[np.isfinite(x_angle)], 0)
    assert np.allclose(y_angle[np.isfinite(y_angle)], 0)
    assert np.allclose(actual_amp, expected_amp)


def test_check_antennas_should_pass_for_matching_antennas():
    _check_antennas(["ANT0", "ANT1"], ["ANT0", "ANT1"])


def test_check_antennas_should_pass_for_absent_axis_with_one_antenna():
    _check_antennas(None, ["ANT0"])


@pytest.mark.parametrize(
    "h5parm_antennas",
    [["ANT1", "ANT0"], ["ANT0", "ANT2"], ["ANT0"]],
    ids=["order", "names", "count"],
)
def test_check_antennas_should_raise_for_mismatching_antennas(
    h5parm_antennas,
):
    with pytest.raises(
        ValueError,
        match="h5parm antennas do not match the visibility antennas",
    ):
        _check_antennas(h5parm_antennas, ["ANT0", "ANT1"])


def test_check_antennas_should_raise_for_absent_axis_with_many_antennas():
    with pytest.raises(
        ValueError,
        match="h5parm has no antenna axis, but visibility has 2 antennas",
    ):
        _check_antennas(None, ["ANT0", "ANT1"])


def _h5parm_gains(
    n_time=2,
    n_ant=3,
    n_freq=4,
    time=True,
    frequency=True,
    antenna=None,
):
    """
    Build the return value of a mocked ``read_h5parm_gains``.

    Parameters
    ----------
    n_time
        Number of solution times.
    n_ant
        Number of antennas.
    n_freq
        Number of solution frequencies.
    time
        Whether the soltab has a time axis.
    frequency
        Whether the soltab has a frequency axis.
    antenna
        Antenna names of the soltab, or None if the axis is absent.

    Returns
    -------
    tuple
        ``(gain, weight, solved, soltab)``.
    """
    shape = (n_time, n_ant, n_freq, 2, 2)
    rng = np.random.default_rng(42)
    gain = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    weight = rng.uniform(size=shape)
    solved = np.ones(shape, dtype=bool)
    soltab = SoltabData(
        values=gain,
        weights=weight,
        solved=solved,
        time=np.arange(n_time, dtype=float) + 100.0 if time else None,
        frequency=(
            np.arange(n_freq, dtype=float) + 1e8 if frequency else None
        ),
        antenna=antenna,
    )
    return gain, weight, solved, soltab


@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "read_h5parm_gains"
)
def test_create_gaintable_from_h5parm_should_create_without_vis(
    read_h5parm_gains_mock,
):
    antennas = ["ANT0", "ANT1", "ANT2"]
    gain, weight, _, soltab = _h5parm_gains(antenna=antennas)
    read_h5parm_gains_mock.return_value = (gain, weight, None, soltab)
    interval = np.array([10.0, 10.0])

    gaintable = create_gaintable_from_h5parm(
        "gains.h5", interval, jones_type="B"
    )

    read_h5parm_gains_mock.assert_called_once_with("gains.h5")
    np.testing.assert_allclose(
        gaintable.CALPARAM_GAIN.data, gain.reshape(*gain.shape[:-2], -1)
    )
    np.testing.assert_allclose(
        gaintable.CALPARAM_WEIGHT.data, weight.reshape(*weight.shape[:-2], -1)
    )
    np.testing.assert_allclose(
        gaintable.CALPARAM_RESIDUAL.data, np.zeros((2, 4, 4))
    )
    np.testing.assert_allclose(gaintable.time.data, soltab.time)
    np.testing.assert_allclose(gaintable.CALPARAM_INTERVAL.data, interval)
    np.testing.assert_allclose(gaintable.frequency.data, soltab.frequency)
    assert gaintable.jones_type == "B"
    assert gaintable.antenna_name.data.tolist() == antennas
    assert gaintable.polarization.data.tolist() == ["XX", "XY", "YX", "YY"]


@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "read_h5parm_gains"
)
def test_create_gaintable_from_h5parm_should_raise_for_absent_antenna_no_vis(
    read_h5parm_gains_mock,
):
    read_h5parm_gains_mock.return_value = _h5parm_gains(antenna=None)

    with pytest.raises(
        ValueError,
        match="H5Parm gains.h5 has no antenna axis; "
        "vis is required to provide the antenna names",
    ):
        create_gaintable_from_h5parm("gains.h5", np.array([10.0, 10.0]))


@pytest.mark.parametrize(
    "time,frequency",
    [(False, True), (True, False), (False, False)],
    ids=["no_time", "no_frequency", "no_time_frequency"],
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "read_h5parm_gains"
)
def test_create_gaintable_from_h5parm_should_raise_for_absent_axis_no_vis(
    read_h5parm_gains_mock, time, frequency
):
    read_h5parm_gains_mock.return_value = _h5parm_gains(
        time=time, frequency=frequency
    )

    with pytest.raises(
        ValueError,
        match="H5Parm gains.h5 has no time or frequency axis; "
        "vis is required to restore them",
    ):
        create_gaintable_from_h5parm("gains.h5", np.array([10.0, 10.0]))


@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "read_h5parm_gains"
)
def test_create_gaintable_from_h5parm_should_use_vis_metadata(
    read_h5parm_gains_mock, generate_vis
):
    vis, _ = generate_vis
    antennas = [str(name) for name in vis.configuration.names.data]
    gain, weight, solved, soltab = _h5parm_gains(
        n_ant=len(antennas), antenna=antennas
    )
    read_h5parm_gains_mock.return_value = (gain, weight, solved, soltab)

    gaintable = create_gaintable_from_h5parm(
        "gains.h5", np.array([10.0, 10.0]), vis=vis, jones_type="G"
    )

    np.testing.assert_allclose(
        gaintable.CALPARAM_GAIN.data, gain.reshape(*gain.shape[:-2], -1)
    )
    np.testing.assert_allclose(gaintable.time.data, soltab.time)
    np.testing.assert_allclose(gaintable.frequency.data, soltab.frequency)
    assert gaintable.jones_type == "G"
    assert gaintable.antenna_name.data.tolist() == antennas
    assert gaintable.polarization.data.tolist() == ["XX", "XY", "YX", "YY"]


@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "read_h5parm_gains"
)
def test_create_gaintable_from_h5parm_should_restore_axes_from_vis(
    read_h5parm_gains_mock, generate_vis
):
    vis, _ = generate_vis
    antennas = [str(name) for name in vis.configuration.names.data]
    read_h5parm_gains_mock.return_value = _h5parm_gains(
        n_time=1,
        n_ant=len(antennas),
        n_freq=1,
        time=False,
        frequency=False,
        antenna=antennas,
    )

    gaintable = create_gaintable_from_h5parm(
        "gains.h5", np.array([10.0]), vis=vis
    )

    np.testing.assert_allclose(
        gaintable.time.data, np.mean(vis.time.data, keepdims=True)
    )
    np.testing.assert_allclose(
        gaintable.frequency.data, np.mean(vis.frequency.data, keepdims=True)
    )
    assert gaintable.CALPARAM_RESIDUAL.shape == (1, 1, 4)


@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "_check_antennas"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers.gaintable."
    "read_h5parm_gains"
)
def test_create_gaintable_from_h5parm_should_check_antennas_against_vis(
    read_h5parm_gains_mock, check_antennas_mock, generate_vis
):
    vis, _ = generate_vis
    antennas = [str(name) for name in vis.configuration.names.data]
    read_h5parm_gains_mock.return_value = _h5parm_gains(
        n_ant=len(antennas), antenna=antennas[::-1]
    )
    check_antennas_mock.side_effect = ValueError("antenna mismatch")

    with pytest.raises(ValueError, match="antenna mismatch"):
        create_gaintable_from_h5parm(
            "gains.h5", np.array([10.0, 10.0]), vis=vis
        )

    check_antennas_mock.assert_called_once_with(antennas[::-1], antennas)
