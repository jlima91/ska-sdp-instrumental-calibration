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
    args : tuple
        Positional arguments for the function.
    kwargs : dict
        Keyword arguments for the function.
    """

    def __init__(self, func, *args, **kwargs):
        """
        Initialize a DeferredTask instance.

        Parameters
        ----------
        func : callable
            The function to be deferred.
        args : tuple
            Positional arguments for the function.
        kwargs : dict
            Keyword arguments for the function.
        """
        lazy_args, args_repack = unpack_collections(args)
        lazy_kwargs, kwargs_repack = unpack_collections(kwargs)

        self.__func = func
        self.__args = args
        self.__kwargs = kwargs

        self.__lazy_args = lazy_args
        self.__lazy_kwargs = lazy_kwargs

        self.__r_args = args_repack
        self.__r_kwarg = kwargs_repack
        self._token = tokenize(func, args, kwargs)

    @property
    def params(self):
        return {
            "args": self.__lazy_args,
            "kwargs": self.__lazy_kwargs,
        }

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

    def __call__(self):
        """
        Execute the deferred function with its arguments.

        Returns
        -------
        Any
            The result of the deferred function.
        """
        return self.__func(*self.__args, **self.__kwargs)

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
        args = self.__r_args(args)[0]
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

    def register(self, func: Callable, *args, **kwargs) -> DeferredTask:
        """
        Register a function as a deferred task.

        Parameters
        ----------
        func : Callable
            The function to defer.
        *args : tuple
            Positional arguments for the function.
        **kwargs : dict
            Keyword arguments for the function.

        Returns
        -------
        DeferredTask
            The registered deferred task.
        """

        deferred_task = DeferredTask(func, *args, **kwargs)

        self._tracked_arrays[deferred_task] = deferred_task.params

        return deferred_task
