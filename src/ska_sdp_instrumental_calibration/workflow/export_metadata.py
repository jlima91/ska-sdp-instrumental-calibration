import logging
import os

import dask
from ska_sdp_dataproduct_metadata import MetaData, ObsCore

logger = logging.getLogger()


def export_metadata_file(metadata_file_path, data_products=[]):
    """
    Writes the metadata to an output YAML file.

    Parameters
    ----------
        metadata_file_path : str
            Path to metadata file.
        data_products : list
            List of INST dataproducts.
    Returns
    -------
        Writes metadata to YAML file.
    """
    if (
        os.environ.get("EXECUTION_BLOCK") is None
        or os.environ.get("PROCESSING_BLOCK") is None
    ):
        logger.warning('Missing "EXECUTION_BLOCK" and "PROCESSING_BLOCK"')
        logger.warning("Could not create metadata file.")
        return dask.delayed(None)
    else:
        metadata_config = {
            "cmdline": None,
            "commit": None,
            "image": os.environ.get("IMAGE"),
            "processing_block": os.environ.get("PROCESSING_BLOCK"),
            "processing_script": os.environ.get("PROCESSING_SCRIPT"),
            "version": os.environ.get("SDP_SCRIPT_VERSION"),
        }
        eb_id = os.environ.get("EXECUTION_BLOCK")
        return export_metadata(
            metadata_file_path, eb_id, metadata_config, data_products
        )


@dask.delayed
def export_metadata(
    metadata_file_path, eb_id, metadata_config, data_products=[]
):
    """
    Generates metadata file for the INST pipeline

    Parameters
    -----------
    metadata_file_path : str
        Path to metadata file.
    eb_id : str
        Execution block ID.
    metadata_config: dict
        Configurations of metadata.
    data_products : list
        List of INST dataproducts.
    """

    if os.path.exists(metadata_file_path):
        metadata = MetaData(path=metadata_file_path)
    else:
        metadata = MetaData()

    metadata.output_path = metadata_file_path
    metadata.set_execution_block_id(eb_id)

    data = metadata.get_data()
    data.config = {**data.config, **metadata_config}
    data.obscore.dataproduct_type = ObsCore.DataProductType.UNKNOWN
    data.obscore.calib_level = ObsCore.CalibrationLevel.LEVEL_0
    data.obscore.obs_collection = (
        f"{ObsCore.SKA}/"
        f"{ObsCore.SKA_LOW}/"
        f"{ObsCore.DataProductType.UNKNOWN}"
    )
    data.obscore.access_format = ObsCore.AccessFormat.UNKNOWN
    data.obscore.facility_name = ObsCore.SKA
    data.obscore.instrument_name = ObsCore.SKA_LOW

    for data_product in data_products:
        metadata.new_file(
            dp_path=data_product["dp_path"],
            description=data_product["description"],
        ).update_status("done")

    metadata.write()
