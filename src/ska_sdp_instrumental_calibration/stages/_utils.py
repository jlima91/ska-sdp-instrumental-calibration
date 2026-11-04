import os
from pathlib import Path

import xarray as xr

from ..scheduler import UpstreamOutput

BEAM_NORMALISED_VIS_KEY = "beam_normalised_vis"
BEAM_NORMALISED_MODELVIS_KEY = "beam_normalised_modelvis"


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


def get_initialisation_vis(
    upstream_output: UpstreamOutput,
) -> tuple[xr.Dataset, xr.Dataset, bool]:
    """
    Get the visibilities to use for initial gain solutions.

    Initial solutions (delay calibration and bandpass initialisation) use
    the visibilities and model visibilities normalised at the beam centre
    when the predict stage provides them, as RCAL does for its initial
    diagonal solve. Otherwise the un-normalised visibilities are used.

    Parameters
    ----------
    upstream_output
        Output from the upstream stage.

    Returns
    -------
        Visibilities, model visibilities, and whether they are the beam
        centre normalised ones.
    """
    if (
        BEAM_NORMALISED_VIS_KEY in upstream_output
        and upstream_output[BEAM_NORMALISED_VIS_KEY] is not None
    ):
        return (
            upstream_output[BEAM_NORMALISED_VIS_KEY],
            upstream_output[BEAM_NORMALISED_MODELVIS_KEY],
            True,
        )
    return upstream_output.vis, upstream_output.modelvis, False
