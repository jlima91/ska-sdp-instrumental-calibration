from typing import Any, Protocol

import dask
from dask.delayed import Delayed


class DeferredTask(Protocol):
    @property
    def params(self) -> dict[str, Any]: ...

    def delayed(self, args: Any, kwargs: Any) -> Delayed: ...


class _TaskManager:
    """
    Manages and computes deferred tasks and their Dask collections.
    """

    def __init__(self):
        """
        Initialize the TaskManager.
        """
        self._tracked_arrays: dict[DeferredTask, dict[str, Any]] = {}

    def compute(self):
        """
        Persist tracked arrays and compute all deferred tasks.

        Returns
        -------
        tuple
            The computed results of all deferred tasks.
        """
        self._tracked_arrays = dask.persist(
            self._tracked_arrays, optimize_graph=True
        )[0]

        try:
            results = dask.compute(
                *[
                    task.delayed(**persisted_value)
                    for task, persisted_value in self._tracked_arrays.items()
                ]
            )
        finally:
            self._tracked_arrays.clear()

        return results

    def register(self, task: DeferredTask):
        """
        Register a task object for deferred execution.

        Parameters
        ----------
        task : DeferredTask
            The initialized task object containing params and a delayed method.
        """

        self._tracked_arrays[task] = task.params


task_manager = _TaskManager()
