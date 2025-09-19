#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=inst
#SBATCH --output=slurm-%j-%x.out

set -e

# User set environment variables
: ${INPUT_PATH:?is not set.}
: ${OUTPUT_PATH:?is not set.}
: ${REPORT_PATH:?is not set.}
: ${CODE_PATH:?is not set.}
: ${META_MODULE:?is not set.}

# Set INST pipeline's inputs
PRE_PROCESSED_CALIBRATOR=$INPUT_PATH/pre_processed_calibrator_68s.ms
CALIBRATOR_SKY_MODEL=$INPUT_PATH/sky_model.csv
INST_CONFIG=$CODE_PATH/scripts/benchmark/inst.yml

# Load relevent modules
module load $META_MODULE
INST_MODULES="py-ska-sdp-benchmark-monitor py-ska-sdp-exec-batchlet py-ska-sdp-instrumental-calibration"
module load $INST_MODULES

# Some extra variables
BATCHLET_CONFIG="${OUTPUT_PATH}/inst_batchlet_config.json"
INST_SUB_COMMAND="experimental"
INST_CACHE_DIR=${INST_CACHE_DIR:-$OUTPUT_PATH}
EVERYBEAM_DATADIR=$(module show everybeam | grep CMAKE_PREFIX_PATH | sed 's=.*CMAKE_PREFIX_PATH \(.*\)/\.=\1/share/everybeam=')

# Create output directories
mkdir -p $OUTPUT_PATH $REPORT_PATH

# Generate and store batchlet's config
cat <<EOF > $BATCHLET_CONFIG
{
  "command": [
    "ska-sdp-instrumental-calibration",
    "$INST_SUB_COMMAND",
    "--config",
    "$INST_CONFIG",
    "--output",
    "$OUTPUT_PATH",
    "--set",
    "parameters.predict_vis.lsm_csv_path",
    "$CALIBRATOR_SKY_MODEL",
    "--set",
    "parameters.predict_vis.eb_coeffs",
    "$EVERYBEAM_DATADIR",
    "--set",
    "parameters.load_data.cache_directory",
    "$INST_CACHE_DIR",
    "--input",
    "$PRE_PROCESSED_CALIBRATOR",
    "--no-unique-output-subdir",
    "--with-report"
  ],
  "dask_params": {
    "threads_per_worker": 4,
    "memory_per_worker": "48GB",
    "resources_per_worker": "process=1",
    "worker_scratch_directory": "$OUTPUT_PATH",
    "use_entry_node": true,
    "dask_cli_option": "--dask-scheduler"
  },
  "monitor": {
    "resources": {
      "level": 0,
      "save_dir": "$REPORT_PATH"
    }
  }
}
EOF

set +e

# Run INST pipeline via batchlet
time batchlet run $BATCHLET_CONFIG

# Smoke test which removes monitor output if pipeline failed
# TODO: Can be moved inside batchlet
PIPELINE_EXIT_CODE=$?
[[ $PIPELINE_EXIT_CODE -ne 0 ]] && \
echo -e "\ninst.sh: Removing monitoring plots because the pipeline failed with exit code $PIPELINE_EXIT_CODE.\n" 1>&2 && \
rm -rf $REPORT_PATH && \
exit $PIPELINE_EXIT_CODE
