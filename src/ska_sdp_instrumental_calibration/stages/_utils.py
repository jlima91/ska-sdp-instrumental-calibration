import inspect
import os
from functools import wraps
from pathlib import Path
from typing import Callable

import xarray as xr

from ..scheduler import UpstreamOutput


def _create_path_tree(path: str):
    """
    Creates parents directory tree for the path.

    Parameters
    ----------
    path: str
        Path for which to create parents directories.
    """
    path_prefix = Path(path)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)


def get_gaintables_path(output_dir: str, file_prefix: str) -> str:
    """
    Obtain path to store gaintables.

    Parameters
    ----------
    output_dir: str
        Directory path where to create gaintables sub directory.
    file_prefix: str
        Plot file prefix.

    Returns
    -------
    str
        Path to store gaintables with file prefix.
    """
    gaintables_path = os.path.join(output_dir, "gaintables", file_prefix)
    _create_path_tree(gaintables_path)
    return gaintables_path


def get_visibilities_path(output_dir: str, file_prefix: str) -> str:
    """
    Obtain path to store visibilities.

    Parameters
    ----------
    output_dir: str
        Directory path where to create visibilities sub directory.
    file_prefix: str
        Plot file prefix.

    Returns
    -------
    str
        Path to store visibilities with file prefix.
    """
    visibilities_path = os.path.join(output_dir, "visibilities", file_prefix)
    _create_path_tree(visibilities_path)
    return visibilities_path


def get_plots_path(output_dir: str, file_prefix: str) -> str:
    """
    Obtain path to store plots.

    Parameters
    ----------
    output_dir: str
        Directory path where to create plots sub directory.
    file_prefix: str
        Plot file prefix.

    Returns
    -------
    str
        Path to store plots with file prefix.
    """
    plots_path = os.path.join(output_dir, "plots", file_prefix)
    _create_path_tree(plots_path)
    return plots_path


def concat_gaintables(upstream_outputs: list[UpstreamOutput]):
    gaintables = [output.gaintable for output in upstream_outputs]
    upstream_output = upstream_outputs[0]
    upstream_output.gaintable = xr.concat(gaintables, dim="time")

    return upstream_output


def fan_out(target_param: str):
    """
    Distributes a scalar function over an iterable for a specific parameter.
    Returns a list of results.
    """

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            items = bound_args.arguments[target_param]

            if not isinstance(items, (list, tuple)):
                return func(*args, **kwargs)

            results = []
            for item in items:
                current_args = bound_args.arguments.copy()
                current_args[target_param] = item
                results.append(func(**current_args))

            return results

        # Update metadata
        wrapper.__metadata__ = {"type": "fan_out"}
        return wrapper

    return decorator


def fan_in(target_param: str, collect_func: Callable):
    """
    Aggregates a list of inputs into a single value before
    passing it to the decorated function.
    """

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            items = bound_args.arguments.get(target_param)

            if not isinstance(items, (list, tuple)):
                return func(*args, **kwargs)

            aggregated_value = collect_func(items)

            current_args = bound_args.arguments.copy()
            current_args[target_param] = aggregated_value

            return func(**current_args)

        wrapper.__metadata__ = {"type": "fan_in"}

        return wrapper

    return decorator
