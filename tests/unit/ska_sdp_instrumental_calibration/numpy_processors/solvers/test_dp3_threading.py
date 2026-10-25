import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.numpy_processors.solvers import (
    dp3_threading,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_threading import (  # noqa: E501
    default_dp3_n_threads,
    get_dp3_lock,
    is_dp3_available,
    require_dp3,
    set_dp3_n_threads,
)

_LOCK_ATTR = dp3_threading._DP3_LOCK_ATTR
_N_THREADS_SET_ATTR = dp3_threading._DP3_N_THREADS_SET_ATTR


@pytest.fixture
def dp3_module():
    """The dp3 module, imported lazily as it is an optional dependency."""
    import dp3

    return dp3


@pytest.fixture
def clean_dp3_lock(dp3_module, monkeypatch):
    """Remove any lock stored on the dp3 module, restoring it afterwards."""
    monkeypatch.delitem(dp3_module.__dict__, _LOCK_ATTR, raising=False)


@pytest.fixture
def clear_dp3_available_cache():
    """Clear the cached dp3 availability before and after the test."""
    is_dp3_available.cache_clear()
    yield
    is_dp3_available.cache_clear()


def test_is_dp3_available_should_return_true_if_dp3_imports(
    clear_dp3_available_cache, monkeypatch
):
    monkeypatch.setitem(sys.modules, "dp3", MagicMock())

    assert is_dp3_available() is True


def test_is_dp3_available_should_return_false_if_dp3_import_fails(
    clear_dp3_available_cache, monkeypatch
):
    # A None entry in sys.modules makes `import dp3` raise ImportError
    monkeypatch.setitem(sys.modules, "dp3", None)

    assert is_dp3_available() is False


def test_is_dp3_available_should_cache_result(
    clear_dp3_available_cache, monkeypatch
):
    monkeypatch.setitem(sys.modules, "dp3", None)
    assert is_dp3_available() is False

    monkeypatch.setitem(sys.modules, "dp3", MagicMock())
    assert is_dp3_available() is False


@patch(
    "ska_sdp_instrumental_calibration.numpy_processors.solvers."
    "dp3_threading.is_dp3_available",
    return_value=True,
)
def test_require_dp3_should_pass_if_dp3_available(is_dp3_available_mock):
    require_dp3("some feature")

    is_dp3_available_mock.assert_called_once_with()


@patch(
    "ska_sdp_instrumental_calibration.numpy_processors.solvers."
    "dp3_threading.is_dp3_available",
    return_value=False,
)
def test_require_dp3_should_raise_if_dp3_unavailable(_):
    with pytest.raises(
        ImportError,
        match="some feature requires the optional 'dp3' package",
    ):
        require_dp3("some feature")


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_create_lock_on_dp3_module(
    dp3_module, clean_dp3_lock
):
    lock = get_dp3_lock()

    assert isinstance(lock, type(threading.RLock()))
    assert dp3_module.__dict__[_LOCK_ATTR] is lock


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_return_same_lock_on_every_call(clean_dp3_lock):
    assert get_dp3_lock() is get_dp3_lock()


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_reuse_existing_lock_on_dp3_module(
    dp3_module, monkeypatch
):
    existing_lock = threading.RLock()
    monkeypatch.setitem(dp3_module.__dict__, _LOCK_ATTR, existing_lock)

    assert get_dp3_lock() is existing_lock


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_return_same_lock_across_threads(
    clean_dp3_lock,
):
    with ThreadPoolExecutor(max_workers=8) as executor:
        locks = list(executor.map(lambda _: get_dp3_lock(), range(32)))

    assert all(lock is locks[0] for lock in locks)


@pytest.fixture
def outside_worker():
    """Make ``distributed.get_worker`` report no running worker."""
    with patch(
        "distributed.get_worker",
        side_effect=ValueError("No worker found"),
    ) as get_worker_mock:
        yield get_worker_mock


@pytest.fixture
def cpu_affinity():
    """Make the process appear to be bound to 5 CPUs."""
    with patch(
        "ska_sdp_instrumental_calibration.numpy_processors.solvers."
        "dp3_threading.os.sched_getaffinity",
        return_value={0, 1, 2, 3, 4},
    ) as sched_getaffinity_mock:
        yield sched_getaffinity_mock


@patch("distributed.get_worker")
def test_default_dp3_n_threads_should_return_worker_nthreads(
    get_worker_mock, monkeypatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    get_worker_mock.return_value = MagicMock(state=MagicMock(nthreads=3))

    assert default_dp3_n_threads() == 3
    get_worker_mock.assert_called_once_with()


@pytest.mark.parametrize(
    "omp_num_threads, expected",
    [("2", 2), (" 6 ", 6), ("4,2", 4)],
)
def test_default_dp3_n_threads_should_use_omp_num_threads_outside_worker(
    outside_worker, cpu_affinity, monkeypatch, omp_num_threads, expected
):
    monkeypatch.setenv("OMP_NUM_THREADS", omp_num_threads)

    assert default_dp3_n_threads() == expected
    cpu_affinity.assert_not_called()


@pytest.mark.parametrize("omp_num_threads", [None, "", "0", "-1", "abc"])
def test_default_dp3_n_threads_should_return_cpu_affinity_outside_worker(
    outside_worker, cpu_affinity, monkeypatch, omp_num_threads
):
    if omp_num_threads is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", omp_num_threads)

    assert default_dp3_n_threads() == 5
    cpu_affinity.assert_called_once_with(0)


@pytest.fixture
def mock_dp3_module(monkeypatch):
    """
    Replace the ``dp3`` package and :py:func:`get_dp3_lock` with mocks.

    Both are children of a single ``manager`` mock, so the order of the
    calls across them can be asserted.
    """
    manager = MagicMock(name="manager")
    monkeypatch.setitem(sys.modules, "dp3", manager.dp3)
    monkeypatch.setattr(dp3_threading, "get_dp3_lock", manager.get_dp3_lock)
    return manager


def test_set_dp3_n_threads_should_set_threads_on_first_call(
    mock_dp3_module,
):
    # DP3 reports a size of 1 before the pool is used
    mock_dp3_module.dp3.get_n_threads.return_value = 1

    set_dp3_n_threads(1)

    mock_dp3_module.dp3.set_n_threads.assert_called_once_with(1)
    assert mock_dp3_module.dp3.__dict__[_N_THREADS_SET_ATTR] is True


def test_set_dp3_n_threads_should_set_threads_if_changed(mock_dp3_module):
    mock_dp3_module.dp3.__dict__[_N_THREADS_SET_ATTR] = True
    mock_dp3_module.dp3.get_n_threads.return_value = 1

    set_dp3_n_threads(4)

    mock_dp3_module.dp3.set_n_threads.assert_called_once_with(4)


def test_set_dp3_n_threads_should_skip_if_unchanged(mock_dp3_module):
    mock_dp3_module.dp3.__dict__[_N_THREADS_SET_ATTR] = True
    mock_dp3_module.dp3.get_n_threads.return_value = 4

    set_dp3_n_threads(4)

    mock_dp3_module.dp3.set_n_threads.assert_not_called()


def test_set_dp3_n_threads_should_hold_lock(mock_dp3_module):
    mock_dp3_module.dp3.__dict__[_N_THREADS_SET_ATTR] = True
    mock_dp3_module.dp3.get_n_threads.return_value = 1

    set_dp3_n_threads(4)

    assert mock_dp3_module.mock_calls == [
        call.get_dp3_lock(),
        call.get_dp3_lock().__enter__(),
        call.dp3.get_n_threads(),
        call.dp3.set_n_threads(4),
        call.get_dp3_lock().__exit__(None, None, None),
    ]


@pytest.mark.requires_dp3
def test_set_dp3_n_threads_should_not_block_if_lock_already_held(
    dp3_module, clean_dp3_lock
):
    n_threads = dp3_module.get_n_threads()

    with get_dp3_lock():
        set_dp3_n_threads(n_threads + 1)
        set_dp3_n_threads(n_threads)

    assert dp3_module.get_n_threads() == n_threads
