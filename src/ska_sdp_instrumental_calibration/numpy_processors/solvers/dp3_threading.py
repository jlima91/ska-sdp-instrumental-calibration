import functools
import os
import threading

__all__ = [
    "is_dp3_available",
    "require_dp3",
    "get_dp3_lock",
    "default_dp3_n_threads",
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


# DP3 is not safe to drive from multiple threads of one process:
# the gaincal solver uses DP3's process-wide (aocommon) thread pool, and
# the bundled HDF5 used to write the h5parm is built without threadsafety.
# Concurrent steps crash the process (std::bad_function_call / SIGABRT),
# e.g. on a dask worker with threads_per_worker > 1.
_DP3_LOCK_ATTR = "_inst_process_lock"


def get_dp3_lock() -> threading.Lock:
    """
    Get the process-wide lock used to serialise DP3 calls.

    The lock is stored on the ``dp3`` module rather than as a global of
    this module, so that functions shipped to dask workers by value
    (cloudpickle) do not try to pickle a lock, and every worker process
    shares a single lock across its threads.

    Returns
    -------
    threading.Lock
        Lock shared by all threads of the current process.
    """
    import dp3  # pylint: disable=import-error,import-outside-toplevel

    # dict.setdefault is atomic under the GIL
    return dp3.__dict__.setdefault(_DP3_LOCK_ATTR, threading.Lock())


def default_dp3_n_threads() -> int:
    """
    Default number of threads for DP3's internal thread pool.

    DP3 sizes its thread pool from the CPU affinity of the process, and
    ignores ``OMP_NUM_THREADS`` and threadpoolctl. On a dask worker this
    would make every worker use all cores, so the pool is limited to the
    number of threads of the worker instead. As DP3 calls are serialised
    by :py:func:`get_dp3_lock`, a single DP3 call can use all of them.

    Returns
    -------
    int
        Number of threads of the current dask worker, or the number of
        CPUs available to the process when not running on a worker.
    """
    from distributed import (  # pylint: disable=import-outside-toplevel
        get_worker,
    )

    try:
        return get_worker().state.nthreads
    except ValueError:
        return len(os.sched_getaffinity(0))
