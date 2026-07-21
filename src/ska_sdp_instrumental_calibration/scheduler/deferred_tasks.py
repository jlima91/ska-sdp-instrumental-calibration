from functools import wraps
from typing import Any, Callable

import dask
from dask.delayed import Delayed


class DeferredTask:
    """
    A class representing a delayed task for execution.

    Parameters
    ----------
    func : callable
        The function to be deferred.
    *args : tuple
        Positional arguments for the function.
    **kwargs : dict
        Keyword arguments for the function.
    """

    def __init__(self, func, *args, **kwargs):
        self.__func = func
        self.__args = args
        self.__kwargs = kwargs

    def __call__(
        self,
        get_persist_args: Callable[[Any], Any],
    ) -> Delayed:
        """
        Executes the deferred task using the provided callable.

        Parameters
        ----------
        get_persist_args : Callable
            Callable to retrieve persisted arguments.

        Returns
        -------
        Delayed
            The delayed function execution object.
        """
        args = get_persist_args(self.__args)
        kwargs = get_persist_args(self.__kwargs)
        return dask.delayed(self.__func)(*args, **kwargs)


class TaskManager:
    """
    Manages and computes deferred tasks and their Dask collections.
    """

    def __init__(self):
        """
        Initializes the TaskManager.
        """
        self._deferred_tasks = []
        self._tracked_arrays = {}

    def _register_lazy_param(self, obj):
        """
        Recursively registers Dask collections for tracking.

        Parameters
        ----------
        obj : Any
            The object or collection to inspect and register.
        """
        if dask.is_dask_collection(obj):
            self._tracked_arrays[id(obj)] = obj
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                self._register_lazy_param(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._register_lazy_param(value)

    def _get_persist_args(self, obj):
        """
        Retrieves arguments from the tracked array.

        Parameters
        ----------
        obj : Any
            The object or collection to process.
        Returns
        -------
        Any
            The arguments reference from the tracked array.
        """
        if id(obj) in self._tracked_arrays:
            return self._tracked_arrays[id(obj)]
        elif isinstance(obj, list):
            return [self._get_persist_args(item) for item in obj]
        elif isinstance(obj, tuple):
            transformed = [self._get_persist_args(item) for item in obj]
            if hasattr(obj, "_fields"):
                return type(obj)(*transformed)
            return tuple(transformed)
        elif isinstance(obj, dict):
            return {k: self._get_persist_args(v) for k, v in obj.items()}
        return obj

    def compute(self):
        """
        Persists tracked arrays and computes all deferred tasks.

        Returns
        -------
        tuple
            The computed results of all deferred tasks.
        """
        self._tracked_arrays = dask.persist(self._tracked_arrays)[0]
        results = dask.compute(
            *[task(self._get_persist_args) for task in self._deferred_tasks]
        )

        self._deferred_tasks.clear()
        self._tracked_arrays.clear()

        return results

    def delayed(self, func: Callable) -> Callable:
        """
        Decorator to register a function as a deferred task.

        Parameters
        ----------
        func : callable
            The function to defer.

        Returns
        -------
        callable
            The wrapped function returning a DeferredTask.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            deferred_task = DeferredTask(func, *args, **kwargs)
            self._deferred_tasks.append(deferred_task)
            self._register_lazy_param(args)
            self._register_lazy_param(kwargs)
            return deferred_task

        return wrapper


task_manager = TaskManager()
