#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=inst-benchmark
#SBATCH --output=slurm-%j-%x.out

set -euo pipefail

# User set environment variables
: ${INPUT_PATH:?is not set.}
: ${WORK_PATH:?is not set.}
: ${OUTPUT_PATH:?is not set.}
: ${REPORT_PATH:?is not set.}
: ${CODE_PATH:?is not set.}
: ${META_MODULE:?is not set.}

# Assumes that there's only one *.ms file in the INPUT_PATH
PRE_PROCESSED_CALIBRATOR=`find "$INPUT_PATH/" -maxdepth 1 -name "*.ms" -print -quit`
# Assumes that there's only one *.csv file in the INPUT_PATH
CALIBRATOR_SKY_MODEL=`find "$INPUT_PATH/" -maxdepth 1 -name "*.csv" -print -quit`

# set dask config
export DASK_CONFIG=$(realpath 'dask_custom_config.yaml')

cat <<EOF > $DASK_CONFIG
distributed:
  comm:
    timeouts:
      connect: '600s'
      tcp: '900s'
  scheduler:
    worker-ttl: '15 minutes'
EOF

# Load relevent modules
module load $META_MODULE
INST_MODULES="py-ska-sdp-benchmark-monitor py-ska-sdp-exec-batchlet py-ska-sdp-instrumental-calibration"
module load $INST_MODULES

# Some extra variables
INST_CONFIG="${CODE_PATH}/scripts/benchmark/inst.yml"
BATCHLET_CONFIG="${OUTPUT_PATH}/inst_batchlet_config.json"
INST_CACHE_DIR=${INST_CACHE_DIR:-$OUTPUT_PATH}

# Create output directories
mkdir -p $OUTPUT_PATH $REPORT_PATH

# Generate and store batchlet's config
cat <<EOF > $BATCHLET_CONFIG
{
  "command": [
    "ska-sdp-instrumental-calibration",
    "experimental",
    "--config",
    "$INST_CONFIG",
    "--output",
    "$OUTPUT_PATH",
    "--set",
    "parameters.predict_vis.lsm_csv_path",
    "$CALIBRATOR_SKY_MODEL",
    "--set",
    "parameters.load_data.cache_directory",
    "$WORK_PATH",
    "--input",
    "$PRE_PROCESSED_CALIBRATOR",
    "--no-unique-output-subdir"
  ],
  "dask_params": {
    "threads_per_worker": 4,
    "memory_per_worker": "48GB",
    "resources_per_worker": "process=1",
    "worker_scratch_directory": "$WORK_PATH",
    "use_entry_node": true,
    "dask_cli_option": "--dask-scheduler",
    "dask_report_dir": "$REPORT_PATH"
  },
  "monitor": {
    "resources": {
      "level": 0,
      "save_dir": "$REPORT_PATH"
    }
  },
  "generate_reports_on_failure": false
}
EOF

# Run INST pipeline via batchlet
time batchlet run $BATCHLET_CONFIG
