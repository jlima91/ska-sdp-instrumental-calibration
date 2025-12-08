from functools import reduce
import os
import sys
from typing import Type

import pandas as pd

from collections.abc import Iterable

module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{module_dir}/../src")

from ska_sdp_piper.piper.configurations.nested_config import NestedConfigParam
from ska_sdp_instrumental_calibration.instrumental_calibration import ska_sdp_instrumental_calibration
from ska_sdp_instrumental_calibration.target_calibration import ska_sdp_instrumental_target_calibration, ska_sdp_instrumental_target_ionospheric_calibration

NONE_FILL = "``null``"

############################################
# Creating dictionary of dataframes
# Each dataframe contains Configuration info
############################################


def process_config_param(prefix, config_param):
    if config_param._type is NestedConfigParam:
        return reduce(
            lambda acc, param: [
                *acc,
                *process_config_param(f"{prefix}.{param[0]}", param[1]),
            ],
            config_param._config_params.items(),
            [],
        )

    return [{"param": prefix, **config_param.__dict__}]


def convert_type_to_string(tp: Type | Iterable[Type]):
    if isinstance(tp, Iterable):
        return [t.__name__ for t in tp]
    return tp.__name__


def generate_config_dfs_per_stage(pipeline_definition):

    dataframes = {}

    for stage in pipeline_definition._stages:
        df = []
        for name, config_param in stage._Stage__config._config_params.items():
            df.extend(process_config_param(name, config_param))

        df = pd.DataFrame(df)
        if df.empty:
            continue

        df = df.rename(columns={"_type": "type"})
        df = df.rename(columns={"_ConfigParam__value": "default"})
        df = df.rename(columns={"allowed_values": "allowed values"})
        df.columns = df.columns.str.capitalize()
        df["Type"] = df["Type"].apply(convert_type_to_string)
        df["Allowed values"] = df["Allowed values"].apply(
            lambda value: "" if value is None else [NONE_FILL if x is None else x for x in value]
        )
        df = df.fillna(NONE_FILL)

        dataframes[stage.name] = df

    return dataframes


calibrator_df = generate_config_dfs_per_stage(ska_sdp_instrumental_calibration)
target_calibration_df = generate_config_dfs_per_stage(ska_sdp_instrumental_target_calibration)
target_ionospeheric_df = generate_config_dfs_per_stage(ska_sdp_instrumental_target_ionospheric_calibration)

#######################
# Generate the RST file
#######################

header = """Stages and configurations
#########################

.. This page is generated using docs/generate_config.py

The descriptions of each stage are copied from the docstrings of stages.
Refer to the `API page for stages <package/guide.html#stages>`_

Each stage has parameters, which are defined in the YAML config file passed to the pipeline.

Instrumental Calibration Stages
*******************************

This section describes the stages used in the Instrumental Calibration pipeline.

{calibrator_stages}

Target Calibration Stages
*************************

This section describes the stages used in the Target Calibration pipeline.

{target_stages}

Target Ionospheric Calibration Stages
*************************

This section describes the stages used in the Target Ionospheric pipeline.

{target_ionospheric_stages}
"""

table_config = """
Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10
"""

indent = "    "


def generate_stage_config(pipeline_definition, dataframes):

    # Write the header first
    output_string = ""
    for stage in pipeline_definition._stages:
        name = stage.name
        df = dataframes[name]
        # Assuming that all stages have "Parameters" section
        doc = stage.__doc__.split(sep="Parameters")[0].rstrip()

        output_string += f"{name}\n{'=' * len(name)}\n{doc}\n{table_config}\n"

        # Convert DataFrame to markdown string and write it to file
        markdown = df.to_markdown(
            index=False,
            tablefmt="grid",
            colalign=["left"] * len(df.columns),
            maxcolwidths=[None, None, 40, 80],
        )
        indented_markdown = "\n".join(
            indent + line for line in markdown.splitlines()
        )

        output_string += f"{indented_markdown}\n\n\n"
    return output_string


out_rst_path = os.path.join(module_dir, "src/stage_config.rst")

with open(out_rst_path, "w") as f:
    # Write the header first
    output_string = f"{header}\n\n"
    calibrator_config = generate_stage_config(
        ska_sdp_instrumental_calibration,
        calibrator_df
    )

    target_config = generate_stage_config(
        ska_sdp_instrumental_target_calibration,
        target_calibration_df
    )

    target_ionospheric_config = generate_stage_config(
        ska_sdp_instrumental_target_ionospheric_calibration,
        target_ionospeheric_df
    )

    output_string = output_string.format(
        calibrator_stages=calibrator_config,
        target_stages=target_config,
        target_ionospheric_stages=target_ionospheric_config
    )
    f.write(output_string)
