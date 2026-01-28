import numpy as np

from ska_sdp_instrumental_calibration.data_managers.solution_interval import (
    SolutionIntervals,
)


def test_should_create_default_solution_intervals():
    time = np.arange(100, 110)
    sol_int = SolutionIntervals(time)

    expected_indices = [
        slice(a, b, 1) for a, b in zip(range(10), range(1, 11))
    ]

    assert sol_int.size == 10

    assert sol_int.indices == expected_indices
    np.testing.assert_allclose(
        sol_int.intervals, np.array([0.9] * 10), rtol=1e-1
    )
    np.testing.assert_allclose(sol_int.solution_time, time)


def test_should_create_full_solution_intervals():
    time = np.arange(100, 110)
    sol_int = SolutionIntervals(time, timeslice="full")

    assert sol_int.size == 1

    assert sol_int.indices == [slice(0, 10, 1)]
    np.testing.assert_allclose(sol_int.intervals, np.array([10]), rtol=1e-1)
    np.testing.assert_allclose(
        sol_int.solution_time, np.array([np.mean(time)])
    )


def test_should_create_partial_solution_intervals():
    time = np.arange(1, 10)
    sol_int = SolutionIntervals(time, timeslice=5)

    assert sol_int.size == 2

    assert sol_int.indices == [slice(0, 5, 1), slice(5, 9, 1)]
    np.testing.assert_allclose(sol_int.intervals, np.array([4, 4]), rtol=1e-1)
    np.testing.assert_allclose(
        sol_int.solution_time, np.array([np.mean(time[:5]), np.mean(time[5:])])
    )
