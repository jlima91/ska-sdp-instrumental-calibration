import logging
import os
from typing import Annotated, Literal

import dask
import xarray as xr
from pydantic import Field
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import (
    INSTMetaData,
    export_gaintable_to_h5parm,
)
from ..scheduler import UpstreamOutput
from ..prism import Prism

logger = logging.getLogger()

INST_METADATA_FILE = "ska-data-product.yaml"


def concat_gaintables(upstream_outputs: list[UpstreamOutput]):
    gaintables = [output.gaintable for output in upstream_outputs]
    upstream_output = upstream_outputs[0]
    upstream_output.gaintable = xr.concat(gaintables, dim="time")

    return upstream_output


@ConfigurableStage(name="export_gain_table")
@Prism.AGGREGATOR
def export_gaintable_stage(
    _upstream_output_: list[UpstreamOutput],
    _output_dir_,
    file_name: Annotated[
        str,
        Field(
            description="""Gain table file name without extension""",
        ),
    ] = "inst.gaintable",
    export_format: Annotated[
        Literal["h5parm", "hdf5"],
        Field(
            description="""Export file format""",
        ),
    ] = "h5parm",
    export_metadata: Annotated[
        bool,
        Field(
            description="""Export metadata into YAML file""",
        ),
    ] = False,
):
    """
    Export gain table solutions to a file.

    Parameters
    ----------
        _upstream_output_: dict
            Output from the upstream stage
        _output_dir_ : str
            Directory path where the output file will be written.
        file_name : str
            Base name for the output file (without extension).
        export_format : str
            Format to export the gain table 'Hdf5' and 'H5parm'.
        export_metadata : bool
            Export metadata to YAML file.
    Returns
    -------
        dict
            Updated upstream output
    """
    _upstream_output_ = concat_gaintables(_upstream_output_)
    gaintable = _upstream_output_.gaintable
    gaintable_file_path = os.path.join(
        _output_dir_, f"{file_name}.{export_format}"
    )
    export_functions = {
        "h5parm": export_gaintable_to_h5parm,
        "hdf5": export_gaintable_to_hdf5,
    }

    logger.info(f"Writing solutions to {gaintable_file_path}")

    export = dask.delayed(export_functions[export_format])(
        gaintable, gaintable_file_path
    )

    _upstream_output_.add_compute_tasks(export)

    if export_metadata and INSTMetaData.can_create_metadata():
        metadata_file_path = os.path.join(_output_dir_, INST_METADATA_FILE)
        inst_metadata = INSTMetaData(
            metadata_file_path,
            data_products=[
                {
                    "dp_path": f"{file_name}.{export_format}",
                    "description": "Gaintable",
                }
            ],
        )
        _upstream_output_.add_compute_tasks(inst_metadata.export())

    return _upstream_output_
