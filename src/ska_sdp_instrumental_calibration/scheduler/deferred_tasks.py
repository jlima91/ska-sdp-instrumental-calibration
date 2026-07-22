from functools import wraps
from typing import Any, Callable

import dask
from dask.base import tokenize, unpack_collections
from dask.delayed import Delayed


class DeferredTask:
    """
    A class representing a delayed task for execution.

    Parameters
    ----------
    func : callable
        The function to be deferred.
    args_repack : callable
        Function to repack positional arguments.
    kwargs_repack : callable
        Function to repack keyword arguments.
    token : str
        Deterministic token identifying the task.
    """

    def __init__(self, func, args_repack, kwargs_repack, token):
        """
        Initialize a DeferredTask instance.

        Parameters
        ----------
        func : callable
            The function to be deferred.
        args_repack : callable
            Function to repack positional arguments.
        kwargs_repack : callable
            Function to repack keyword arguments.
        token : str
            Deterministic token identifying the task.
        """
        self.__func = func
        self.__r_args = args_repack
        self.__r_kwarg = kwargs_repack
        self._token = token

    def __eq__(self, other):
        """
        Check equality based on the deterministic token.

        Parameters
        ----------
        other : Any
            The object to compare against.

        Returns
        -------
        bool
            True if tokens match, otherwise False.
        """
        if not isinstance(other, DeferredTask):
            return NotImplemented
        return self._token == other._token

    def __hash__(self):
        """
        Compute the hash of the task using its token.

        Returns
        -------
        int
            The hash value of the task's token.
        """
        return hash(self._token)

    def delayed(self, args, kwargs) -> Delayed:
        """
        Create a dask.delayed object for the task.

        Parameters
        ----------
        args : list or tuple
            Unpacked positional arguments.
        kwargs : dict or tuple
            Unpacked keyword arguments.

        Returns
        -------
        Delayed
            The delayed dask computation object.
        """
        args = self.__r_args(args)
        kwargs = self.__r_kwarg(kwargs)[0]

        return dask.delayed(self.__func)(*args, **kwargs)


class TaskManager:
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
        self._tracked_arrays = dask.persist(self._tracked_arrays)[0]

        results = dask.compute(
            *[
                task.delayed(**persisted_value)
                for task, persisted_value in self._tracked_arrays.items()
            ]
        )

        self._tracked_arrays.clear()

        return results

    def register(self, func: Callable) -> Callable:
        """
        Decorator to register a function as a deferred task.

        Parameters
        ----------
        func : Callable
            The function to defer.

        Returns
        -------
        Callable
            The wrapped function returning a DeferredTask.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            lazy_args, args_repack = unpack_collections(*args)
            lazy_kwargs, kwargs_repack = unpack_collections(kwargs)
            token = tokenize(func, args, kwargs)

            deferred_task = DeferredTask(
                func, args_repack, kwargs_repack, token
            )

            self._tracked_arrays[deferred_task] = {
                "args": lazy_args,
                "kwargs": lazy_kwargs,
            }

            return deferred_task

        return wrapper
