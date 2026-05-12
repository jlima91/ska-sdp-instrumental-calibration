import logging
import os
from functools import reduce
from typing import Annotated, Literal, Optional

import dask
import xarray as xr
from pydantic import Field
from ska_sdp_datamodels.calibration.calibration_functions import (
    export_gaintable_to_hdf5,
)
from ska_sdp_piper.piper import CLIArgument, ConfigurableStage

from ..data_managers.data_export import (
    INSTMetaData,
    export_gaintable_to_h5parm,
)
from ..data_managers.sdm import get_gaintable_file_path
from ..scheduler import UpstreamOutput
from ..tagger import Tags

logger = logging.getLogger()

INST_METADATA_FILE = "ska-data-product.yaml"


def concat_gaintables(upstream_outputs: list[UpstreamOutput]):
    if len(upstream_outputs) == 1:
        return upstream_outputs[0]

    gaintables = [output.gaintable for output in upstream_outputs]
    upstream_output = upstream_outputs[0]
    upstream_output.gaintable = xr.concat(gaintables, dim="time")

    return upstream_output


def group_upstream_by_field_id(upstream_outputs):
    def accumulate_by_field_id(acc, upstream):
        if upstream.field_id not in acc:
            acc[upstream.field_id] = []

        acc[upstream.field_id].append(upstream)
        return acc

    return reduce(accumulate_by_field_id, upstream_outputs, {})


@ConfigurableStage(name="export_gain_table")
@Tags.AGGREGATOR
def export_gaintable_stage(
    _upstream_output_: list[UpstreamOutput],
    _output_dir_,
    sdm_path: Annotated[Optional[str], CLIArgument] = None,
    file_name: Annotated[
        str,
        Field(
            description="""Gain table file name without extension""",
        ),
    ] = "gaintable",
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
    final_upstream = UpstreamOutput()
    export_functions = {
        "h5parm": export_gaintable_to_h5parm,
        "hdf5": export_gaintable_to_hdf5,
    }
    grouped_upstream_output = group_upstream_by_field_id(_upstream_output_)
    for field_id, upstream_outputs in grouped_upstream_output.items():
        upstream_output = concat_gaintables(upstream_outputs)

        gaintable = upstream_output.gaintable
        gaintable_filename = f"{file_name}.{export_format}"
        purpose = upstream_output.calibration_purpose
        gaintable_file_path = get_gaintable_file_path(
            output_dir=_output_dir_,
            filename=gaintable_filename,
            sdm_path=sdm_path,
            purpose=purpose,
            field_id=field_id,
        )

        logger.info(f"Writing solutions to {gaintable_file_path}")

        export = dask.delayed(export_functions[export_format])(
            gaintable, gaintable_file_path
        )

        final_upstream.add_compute_tasks(export)

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
        final_upstream.add_compute_tasks(inst_metadata.export())

    return final_upstream
