import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.h5parm import (
    SoltabData,
    combine_amplitude_phase,
)

SHAPE = (1, 2, 3, 2, 2)


def _soltab(value: float, weight: float, solved: np.ndarray) -> SoltabData:
    return SoltabData(
        values=np.full(SHAPE, value),
        weights=np.full(SHAPE, weight),
        solved=np.broadcast_to(solved, SHAPE),
        time=None,
        frequency=None,
        antenna=None,
    )


def test_should_combine_amplitude_and_phase():
    amplitude = _soltab(2.0, 0.5, np.ones((2, 2), dtype=bool))
    phase = _soltab(np.pi / 2, 0.25, np.eye(2, dtype=bool))

    gain, weight, solved = combine_amplitude_phase(amplitude, phase)

    np.testing.assert_allclose(gain, 2j)
    assert np.all(weight == 0.25)
    np.testing.assert_array_equal(solved[0, 0, 0], np.eye(2, dtype=bool))


def test_should_assume_zero_phase_without_phase_soltab():
    amplitude = _soltab(2.0, 0.5, np.ones((2, 2), dtype=bool))

    gain, weight, solved = combine_amplitude_phase(amplitude, None)

    np.testing.assert_allclose(gain, 2.0)
    assert weight is amplitude.weights
    assert solved is amplitude.solved


def test_should_assume_unit_diagonal_amplitude_without_amplitude_soltab():
    phase = _soltab(np.pi / 2, 0.25, np.eye(2, dtype=bool))

    gain, weight, solved = combine_amplitude_phase(None, phase)

    np.testing.assert_allclose(gain[..., 0, 0], 1j)
    np.testing.assert_allclose(gain[..., 1, 1], 1j)
    np.testing.assert_allclose(gain[..., 0, 1], 0)
    np.testing.assert_allclose(gain[..., 1, 0], 0)
    assert weight is phase.weights
    assert solved is phase.solved


def test_should_raise_exception_without_amplitude_and_phase_soltabs():
    with pytest.raises(ValueError, match="Either amplitude or phase"):
        combine_amplitude_phase(None, None)
