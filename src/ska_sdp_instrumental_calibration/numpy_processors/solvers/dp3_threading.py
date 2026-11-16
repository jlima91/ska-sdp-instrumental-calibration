import functools
import os
import threading

__all__ = [
    "is_dp3_available",
    "require_dp3",
    "get_dp3_lock",
    "default_dp3_n_threads",
    "set_dp3_n_threads",
]


@functools.cache
def is_dp3_available() -> bool:
    """
    Check whether the optional ``dp3`` package can be imported.

    ``dp3`` is imported lazily on the first call only, so that importing
    INST does not load DP3. The result is cached.

    Returns
    -------
    bool
        True if ``dp3`` is installed and importable.
    """
    try:
        # pylint: disable-next=import-outside-toplevel,unused-import
        import dp3  # noqa: F401
    except ImportError:
        return False
    return True


def require_dp3(feature: str) -> None:
    """
    Ensure that the optional ``dp3`` package is available.

    Parameters
    ----------
    feature
        Name of the feature requiring DP3, used in the error message.

    Raises
    ------
    ImportError
        If ``dp3`` is not installed or can not be imported.
    """
    if not is_dp3_available():
        raise ImportError(
            f"{feature} requires the optional 'dp3' package. Install it "
            "with: pip install 'ska-sdp-instrumental-calibration[dp3]'"
        )


_DP3_LOCK_ATTR = "_inst_process_lock"
"""
Name of the ``dp3`` module attribute holding the process-wide DP3 lock.

DP3 is not safe to drive from multiple threads of one process: the
gaincal solver uses DP3's process-wide (aocommon) thread pool, and the
bundled HDF5 used to write the h5parm is built without threadsafety.
Concurrent steps crash the process (std::bad_function_call / SIGABRT),
e.g. on a dask worker with threads_per_worker > 1.
"""

_DP3_N_THREADS_SET_ATTR = "_inst_n_threads_set"
"""
Name of the ``dp3`` module attribute marking that the size of DP3's
thread pool was set by :py:func:`set_dp3_n_threads` in this process.

Until the pool is first used, ``dp3.get_n_threads`` reports 1, while the
first ``dp3.make_step`` then sizes the pool to all CPU cores. So the
reported size can only be trusted after the pool was set explicitly.
"""


def get_dp3_lock() -> threading.RLock:
    """
    Get the process-wide lock used to serialise DP3 calls.

    The lock is stored on the ``dp3`` module rather than as a global of
    this module, so that functions shipped to dask workers by value
    (cloudpickle) do not try to pickle a lock, and every worker process
    shares a single lock across its threads.

    The lock is reentrant, so that helpers like
    :py:func:`set_dp3_n_threads` can acquire it on their own and also be
    called by code already holding it.

    Returns
    -------
    threading.RLock
        Lock shared by all threads of the current process.
    """
    import dp3  # pylint: disable=import-error,import-outside-toplevel

    # dict.setdefault is atomic under the GIL
    return dp3.__dict__.setdefault(_DP3_LOCK_ATTR, threading.RLock())


def default_dp3_n_threads() -> int:
    """
    Default number of threads for DP3's internal thread pool.

    DP3 runs from the command line take the size of the thread pool from
    the ``numthreads`` key, else ``OMP_NUM_THREADS``, else the number of
    CPU cores. Steps created through the DP3 Python bindings ignore
    ``OMP_NUM_THREADS`` though, and use all CPU cores unless the size is
    set with ``dp3.set_n_threads``. The size is therefore chosen here,
    aligned with DP3 itself, in order of precedence:

    1. On a dask worker, the number of threads of the worker. As DP3
       calls are serialised by :py:func:`get_dp3_lock`, a single DP3 call
       can use all of them. ``OMP_NUM_THREADS`` is not used here, as the
       dask nanny sets it to 1 for every worker process.
    2. Otherwise, the first value of ``OMP_NUM_THREADS``, if set to a
       positive integer.
    3. Otherwise, the number of CPUs available to the process.

    threadpoolctl limits are not considered: they only apply to OpenMP
    and BLAS runtimes, not to DP3's own thread pool.

    Returns
    -------
    int
        Number of threads for DP3's thread pool.
    """
    from distributed import (  # pylint: disable=import-outside-toplevel
        get_worker,
    )

    try:
        return get_worker().state.nthreads
    except ValueError:
        pass

    omp_n_threads = _omp_num_threads()
    if omp_n_threads is not None:
        return omp_n_threads

    return len(os.sched_getaffinity(0))


def _omp_num_threads() -> int | None:
    """
    Number of threads requested by the ``OMP_NUM_THREADS`` variable.

    ``OMP_NUM_THREADS`` may hold a comma separated list for nested
    parallelism, of which only the first (outermost) value is used.

    Returns
    -------
    int or None
        The requested number of threads, or None if the variable is not
        set or is not a positive integer.
    """
    value = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    try:
        n_threads = int(value)
    except ValueError:
        return None
    return n_threads if n_threads > 0 else None


def set_dp3_n_threads(n_threads: int) -> None:
    """
    Set the size of DP3's process-wide thread pool, if it changed.

    ``dp3.set_n_threads`` stops and restarts the whole thread pool on
    every call, even when the size does not change, so the pool is only
    resized when its current size differs from ``n_threads``. The pool
    is always set on the first call in a process, as DP3 reports a size
    of 1 until then (see :py:data:`_DP3_N_THREADS_SET_ATTR`).

    The check and resize are done while holding the lock from
    :py:func:`get_dp3_lock`, which may already be held by the caller.

    Parameters
    ----------
    n_threads
        Number of threads for DP3's thread pool.
    """
    import dp3  # pylint: disable=import-error,import-outside-toplevel

    with get_dp3_lock():
        if (
            not dp3.__dict__.get(_DP3_N_THREADS_SET_ATTR, False)
            or dp3.get_n_threads() != n_threads
        ):
            dp3.set_n_threads(n_threads)
            dp3.__dict__[_DP3_N_THREADS_SET_ATTR] = True
