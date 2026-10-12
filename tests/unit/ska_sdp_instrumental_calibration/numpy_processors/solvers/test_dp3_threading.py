import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.numpy_processors.solvers import (
    dp3_threading,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_threading import (  # noqa: E501
    default_dp3_n_threads,
    get_dp3_lock,
    is_dp3_available,
    require_dp3,
)

_LOCK_ATTR = dp3_threading._DP3_LOCK_ATTR


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

    assert isinstance(lock, type(threading.Lock()))
    assert dp3_module.__dict__[_LOCK_ATTR] is lock


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_return_same_lock_on_every_call(clean_dp3_lock):
    assert get_dp3_lock() is get_dp3_lock()


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_reuse_existing_lock_on_dp3_module(
    dp3_module, monkeypatch
):
    existing_lock = threading.Lock()
    monkeypatch.setitem(dp3_module.__dict__, _LOCK_ATTR, existing_lock)

    assert get_dp3_lock() is existing_lock


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_return_same_lock_across_threads(
    clean_dp3_lock,
):
    with ThreadPoolExecutor(max_workers=8) as executor:
        locks = list(executor.map(lambda _: get_dp3_lock(), range(32)))

    assert all(lock is locks[0] for lock in locks)


@patch("distributed.get_worker")
def test_default_dp3_n_threads_should_return_worker_nthreads(get_worker_mock):
    get_worker_mock.return_value = MagicMock(state=MagicMock(nthreads=3))

    assert default_dp3_n_threads() == 3
    get_worker_mock.assert_called_once_with()


@patch(
    "ska_sdp_instrumental_calibration.numpy_processors.solvers."
    "dp3_threading.os.sched_getaffinity"
)
@patch("distributed.get_worker")
def test_default_dp3_n_threads_should_return_cpu_affinity_outside_worker(
    get_worker_mock, sched_getaffinity_mock
):
    get_worker_mock.side_effect = ValueError("No worker found")
    sched_getaffinity_mock.return_value = {0, 1, 2, 3, 4}

    assert default_dp3_n_threads() == 5
    sched_getaffinity_mock.assert_called_once_with(0)
