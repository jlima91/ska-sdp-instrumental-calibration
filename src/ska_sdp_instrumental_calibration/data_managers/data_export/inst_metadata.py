import logging
import os

import dask

try:
    from ska_sdp_dataproduct_metadata import MetaData, ObsCore

    metadata_package_available = True
except ImportError:
    metadata_package_available = False

logger = logging.getLogger()


class INSTMetaData:
    @staticmethod
    def can_create_metadata():
        """
        Check if metadata can be created.
        It checks whether EXECUTION_BLOCK_ID and
        PROCESSING_BLOCK_ID are present in the env.

        Returns
        -------
        bool
        """
        eb_id = os.environ.get("EXECUTION_BLOCK_ID")
        pb_id = os.environ.get("PROCESSING_BLOCK_ID")

        return (
            metadata_package_available
            and eb_id is not None
            and pb_id is not None
        )

    def __init__(self, path, data_products=None):
        """
        Generates metadata file for the INST pipeline

        Parameters
        ----------
        path : str
            Path to metadata file.
        data_products : list
            List of INST dataproducts.
        """
        self.__data_products = [] if data_products is None else data_products
        self.__metadata = (
            MetaData(path) if os.path.exists(path) else MetaData()
        )
        self.__metadata.output_path = path

    @dask.delayed
    def export(self):
        """
        Exports INST metadata.
        """
        self.__prepare_metadata()
        self.__metadata.write()

    def __prepare_metadata(self):
        """
        Prepares the metadata.
        """
        config = {
            "cmdline": None,
            "commit": None,
            "image": os.environ.get("PROCESSING_SCRIPT_IMAGE"),
            "processing_block": os.environ.get("PROCESSING_BLOCK_ID"),
            "processing_script": os.environ.get("PROCESSING_SCRIPT_NAME"),
            "version": os.environ.get("PROCESSING_SCRIPT_VERSION"),
        }

        eb_id = os.environ.get("EXECUTION_BLOCK_ID")
        self.__metadata.set_execution_block_id(eb_id)

        data = self.__metadata.get_data()
        data.config = {**data.config, **config}
        data.obscore.dataproduct_type = ObsCore.DataProductType.UNKNOWN
        data.obscore.calib_level = ObsCore.CalibrationLevel.LEVEL_0
        data.obscore.access_format = ObsCore.AccessFormat.HDF5
        data.obscore.facility_name = ObsCore.SKA
        data.obscore.instrument_name = ObsCore.SKA_LOW
        data.obscore.obs_collection = (
            f"{data.obscore.facility_name}/"
            f"{data.obscore.instrument_name}/"
            f"{data.obscore.dataproduct_type}"
        )

        for data_product in self.__data_products:
            self.__metadata.new_file(
                dp_path=data_product["dp_path"],
                description=data_product["description"],
            ).update_status("done")
