from typing import Callable

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

    def __init__(self, func: Callable, *args, **kwargs):
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
        dask_args, args_repack = unpack_collections(args)
        dask_kwargs, kwargs_repack = unpack_collections(kwargs)

        self.__func = func
        self.__args = args
        self.__kwargs = kwargs

        self.__dask_args = dask_args
        self.__dask_kwargs = dask_kwargs

        self.__r_args = args_repack
        self.__r_kwarg = kwargs_repack
        self._token = tokenize(func, args, kwargs)

    @property
    def dask_params(self):
        return {
            "args": self.__dask_args,
            "kwargs": self.__dask_kwargs,
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
