from typing import Iterable

import h5py
import numpy as np
from numpy.typing import NDArray
from ska_sdp_datamodels.calibration.calibration_model import GainTable

from ...logger import setup_logger
from ...workflow.utils import create_soltab_datasets, create_soltab_group

logger = setup_logger("data_managers.data_export")


def _ndarray_of_null_terminated_bytes(strings: Iterable[str]) -> NDArray:
    # NOTE: making antenna names one character longer, in keeping with
    # ska-sdp-batch-preprocess
    return np.asarray([s.encode("ascii") + b"\0" for s in strings])


def export_gaintable_to_h5parm(
    gaintable: GainTable, filename: str, squeeze: bool = False
):
    """Export a GainTable to a H5Parm HDF5 file.

    :param gaintable: GainTable
    :param filename: Name of H5Parm file
    :param squeeze: If True, remove axes of length one from dataset
    """
    logger.info(f"exporting cal solutions to {filename}")

    # check gaintable gain and weight dimensions
    dims = ["time", "antenna", "frequency", "receptor1", "receptor2"]
    if list(gaintable.gain.sizes) != dims:
        raise ValueError(f"Unexpected dims: {list(gaintable.gain.sizes)}")

    # adjust dimensions to be consistent with H5Parm output format
    gaintable = gaintable.rename({"antenna": "ant", "frequency": "freq"})
    gaintable = gaintable.stack(pol=("receptor1", "receptor2"))
    polstrs = _ndarray_of_null_terminated_bytes(
        [f"{p1}{p2}" for p1, p2 in gaintable["pol"].data]
    )
    gaintable = gaintable.assign_coords({"pol": polstrs})

    # check polarisations and discard unused terms
    polstrs = _ndarray_of_null_terminated_bytes(["XX", "XY", "YX", "YY"])
    if not np.array_equal(gaintable["pol"].data, polstrs):
        raise ValueError("Subsequent pipelines assume linear pol order")
    if np.sum(np.abs(gaintable.isel(pol=[1, 2]).weight.data)) == 0:
        gaintable = gaintable.isel(pol=[0, 3])

    # replace antenna indices with antenna names
    if gaintable.configuration is None:
        raise ValueError("Missing gt config. H5Parm requires antenna names")
    antenna_names = _ndarray_of_null_terminated_bytes(
        gaintable.configuration.names.data[gaintable["ant"].data]
    )
    gaintable = gaintable.assign_coords({"ant": antenna_names})

    # remove axes of length one if required
    if squeeze:
        gaintable = gaintable.squeeze(drop=True)

    logger.info(f"output dimensions: {dict(gaintable.gain.sizes)}")

    with h5py.File(filename, "w") as file:

        solset = file.create_group("sol000")

        # Amplitude table
        soltab = create_soltab_group(solset, "amplitude")
        val, weight = create_soltab_datasets(soltab, gaintable)
        val[...] = np.absolute(gaintable["gain"].data)
        weight[...] = gaintable["weight"].data

        # Phase table
        soltab = create_soltab_group(solset, "phase")
        val, weight = create_soltab_datasets(soltab, gaintable)
        val[...] = np.angle(gaintable["gain"].data)
        weight[...] = gaintable["weight"].data
