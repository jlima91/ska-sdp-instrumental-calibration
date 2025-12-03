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


def parse_reference_antenna(refant, gaintable):
    """
    Checks and converts station names

    Parameters
    ----------
        refant: int or str
            Reference antenna.
        gaintable: Gaintable Dataset
            Gaintable
    Returns
    -------
        refant: Reference antenna index
    """
    if type(refant) is str:
        station_names = gaintable.configuration.names
        try:
            station_index = station_names.where(
                station_names == refant, drop=True
            ).id.values[0]
        except IndexError:
            raise ValueError("Reference antenna name is not valid")
        return station_index
    elif type(refant) is int:
        station_count = gaintable.antenna.size
        if refant > station_count - 1 or refant < 0:
            raise ValueError("Reference antenna index is not valid")
        else:
            return refant
