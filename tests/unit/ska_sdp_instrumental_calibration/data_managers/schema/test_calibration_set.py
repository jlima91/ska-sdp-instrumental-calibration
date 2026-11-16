import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.schema.calibration_set import (  # noqa: E501
    GainCalibrationSetXds,
)


def _constructor_kwargs(n_time=2, n_ant=3, n_freq=4, n_pol=4):
    gain_shape = (n_time, n_ant, n_freq, n_pol)
    return dict(
        gain=np.ones(gain_shape, dtype=np.complex64),
        weight=np.ones(gain_shape, dtype=np.float32),
        residual=np.zeros((n_time, n_freq, n_pol), dtype=np.float32),
        time=np.arange(n_time, dtype=float),
        interval=np.ones(n_time, dtype=float),
        frequency=np.arange(n_freq, dtype=float) + 1e8,
        antenna_name=np.array([f"ANT{i}" for i in range(n_ant)]),
        polarization=np.array(["XX", "XY", "YX", "YY"][:n_pol]),
        jones_type="B",
    )


def test_constructor_should_set_coordinates():
    kwargs = _constructor_kwargs()

    gaintable = GainCalibrationSetXds.constructor(**kwargs)

    assert gaintable.antenna_name.data.tolist() == ["ANT0", "ANT1", "ANT2"]
    assert gaintable.polarization.data.tolist() == ["XX", "XY", "YX", "YY"]
    assert gaintable.jones_type == "B"


def test_constructor_should_accept_two_pols():
    kwargs = _constructor_kwargs(n_pol=2)
    kwargs["polarization"] = np.array(["XX", "YY"])

    gaintable = GainCalibrationSetXds.constructor(**kwargs)

    assert gaintable.polarization.data.tolist() == ["XX", "YY"]


def test_constructor_should_raise_for_antenna_name_mismatch():
    kwargs = _constructor_kwargs()
    kwargs["antenna_name"] = np.array(["ANT0", "ANT1"])

    with pytest.raises(
        ValueError, match="Got 2 antenna names for 3 antennas in gain"
    ):
        GainCalibrationSetXds.constructor(**kwargs)


def test_constructor_should_raise_for_polarization_mismatch():
    kwargs = _constructor_kwargs()
    kwargs["polarization"] = np.array(["XX", "YY"])

    with pytest.raises(
        ValueError,
        match="Got 2 polarization labels for 4 polarizations in gain",
    ):
        GainCalibrationSetXds.constructor(**kwargs)
