import logging
import os

import dask
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.data_export import (
    INSTMetaData,
    export_gaintable_to_h5parm,
)

logger = logging.getLogger()

INST_METADATA_FILE = "ska-data-product.yaml"


@ConfigurableStage(
    "export_gain_table",
    configuration=Configuration(
        file_name=ConfigParam(
            str,
            "gaintable",
            description="Gain table file name without extension",
        ),
        export_format=ConfigParam(
            str,
            "h5parm",
            description="Export file format",
            allowed_values=["h5parm", "hdf5"],
        ),
        export_metadata=ConfigParam(
            bool,
            False,
            description="Export metadata into YAML file",
        ),
    ),
)
def export_gaintable_stage(
    upstream_output, file_name, export_format, export_metadata, _output_dir_
):
    """
    Export gain table solutions to a file.

    Parameters
    ----------
        upstream_output : dict
            Output from the upstream stage.
        file_name : str
            Base name for the output file (without extension).
        export_format : str
            Format to export the gain table 'Hdf5' and 'H5parm'.
        export_metadata : bool
            Export metadata to YAML file.
        _output_dir_ : str
            Directory path where the output file will be written.
    Returns
    -------
        dict
            Updated upstream output
    """
    gaintable = upstream_output.gaintable
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

    upstream_output.add_compute_tasks(export)

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
        upstream_output.add_compute_tasks(inst_metadata.export())

    return upstream_output
