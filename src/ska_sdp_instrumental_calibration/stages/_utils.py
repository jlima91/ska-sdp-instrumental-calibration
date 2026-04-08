import os
from pathlib import Path


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
