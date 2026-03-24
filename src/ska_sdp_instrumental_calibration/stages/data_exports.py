import logging
import os
from typing import Annotated, Literal

import dask
from pydantic import Field
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import (
    INSTMetaData,
    export_gaintable_to_h5parm,
)

logger = logging.getLogger()

INST_METADATA_FILE = "ska-data-product.yaml"


@ConfigurableStage(name="export_gain_table")
def export_gaintable_stage(
    _upstream_output_,
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
