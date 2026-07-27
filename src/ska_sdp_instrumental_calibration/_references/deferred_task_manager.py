from collections import namedtuple
from functools import wraps

import dask
import dask.array as da

# Efficient, immutable container for queued task recipes
TaskRecipe = namedtuple("TaskRecipe", ["func", "args", "kwargs"])


class DeferredTaskManager:
    def __init__(self):
        self._registry: list[TaskRecipe] = list()
        self._tracked_arrays: dict[int, da.Array] = (
            dict()
        )  # mapping of id(lazy_array) -> lazy_array

    def _find_arrays(self, obj):
        """Recursively parses arguments to find all unique Dask/Xarray targets."""
        if dask.is_dask_collection(obj):
            self._tracked_arrays[id(obj)] = obj
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                self._find_arrays(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._find_arrays(value)

    def _replace_with_persisted(self, obj, mapping):
        """Recursively swaps lazy arrays with their cluster-persisted counterparts."""
        if id(obj) in mapping:
            return mapping[id(obj)]
        elif isinstance(obj, list):
            return [
                self._replace_with_persisted(item, mapping) for item in obj
            ]
        elif isinstance(obj, tuple):
            transformed = [
                self._replace_with_persisted(item, mapping) for item in obj
            ]
            # If it's a namedtuple, unpack elements into its original constructor
            if hasattr(obj, "_fields"):
                return type(obj)(*transformed)
            return tuple(transformed)
        elif isinstance(obj, dict):
            return {
                k: self._replace_with_persisted(v, mapping)
                for k, v in obj.items()
            }
        return obj

    def delayed(self, func):
        """Decorator to wrap functions, deferring their execution and cataloging inputs."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.register(
                TaskRecipe(func=func, args=args, kwargs=kwargs)
            )

        return wrapper

    def register(self, task_recipe: TaskRecipe):
        self._find_arrays(task_recipe.args)
        self._find_arrays(task_recipe.kwargs)
        self._registry.append(task_recipe)
        return task_recipe

    def compute(self):
        """Executes the pipeline: Persists arrays first, builds delayed graphs, then computes."""
        if not self._registry:
            return []

        # Dask unpacks the dict, computes arrays chunkwise in parallel,
        # and returns a twin dictionary filled with live cluster futures.
        persisted_mapping = dask.persist(self._tracked_arrays)[0]

        # Construct the true dask.delayed tasks using the persisted mapping
        actual_delayed_tasks = []
        for task in self._registry:
            persisted_args = self._replace_with_persisted(
                task.args, persisted_mapping
            )
            persisted_kwargs = self._replace_with_persisted(
                task.kwargs, persisted_mapping
            )

            actual_delayed = dask.delayed(task.func)(
                *persisted_args, **persisted_kwargs
            )
            actual_delayed_tasks.append(actual_delayed)

        # Launch computation of delayed tasks
        results = dask.compute(*actual_delayed_tasks)

        # Wipe state for clean garbage collection and readiness for the next run
        self._registry.clear()
        self._tracked_arrays.clear()

        return results
