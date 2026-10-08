import numpy as np
import pytest
from mock import patch

from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_solvers import (  # noqa: E501
    Dp3GaincalSolver,
    dp3_gaincal_solve,
)


@pytest.fixture
def dp3_unavailable():
    with patch(
        "ska_sdp_instrumental_calibration.numpy_processors.solvers."
        "dp3_threading.is_dp3_available",
        return_value=False,
    ):
        yield


@patch(
    "ska_sdp_instrumental_calibration.numpy_processors.solvers."
    "dp3_solvers.require_dp3"
)
def test_dp3_gaincal_solver_should_require_dp3(require_dp3_mock):
    Dp3GaincalSolver()

    require_dp3_mock.assert_called_once_with("dp3_gaincal solver")


def test_dp3_gaincal_solver_should_raise_if_dp3_unavailable(dp3_unavailable):
    with pytest.raises(
        ImportError,
        match="dp3_gaincal solver requires the optional 'dp3' package",
    ):
        Dp3GaincalSolver()


def test_dp3_gaincal_solve_should_raise_if_dp3_unavailable(dp3_unavailable):
    vis = np.zeros((1, 1, 1, 4), dtype=complex)
    gain = np.ones((1, 2, 1, 2, 2), dtype=complex)

    with pytest.raises(
        ImportError,
        match="dp3_gaincal_solve requires the optional 'dp3' package",
    ):
        dp3_gaincal_solve(
            vis,
            np.zeros(vis.shape, dtype=bool),
            np.ones(vis.shape),
            vis,
            None,
            gain,
            np.ones(gain.shape),
            np.zeros(gain.shape),
            np.array([0]),
            np.array([1]),
        )
