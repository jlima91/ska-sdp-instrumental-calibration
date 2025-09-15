#!/usr/bin/env bash
 
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=inst
#SBATCH --output=slurm-%j-%x.out

set -e

# User set environment variables
: ${SKA_SDP_MODULE_PATH:?is not set.} # /shared/fsx1/shared/metamodules
: ${SKA_SDP_MODULE_NAME:?is not set.} # ska-sdp-spack/2025.08.3

# Set INST pipeline's inputs
PRE_PROCESSED_CALIBRATOR=/shared/fsx1/dhruva/sp_5859_inst/pre_processed_calibrator_68s.ms
INST_CONFIG=/shared/fsx1/dhruva/sp_5859_inst/inst.yml
CALIBRATOR_SKY_MODEL=/shared/fsx1/dhruva/sp_5859_inst/sky_model.csv

# Load relevent modules
export MODULEPATH="$SKA_SDP_MODULE_PATH":${MODULEPATH}
module load $SKA_SDP_MODULE_NAME

INST_MODULES="py-ska-sdp-benchmark-monitor py-ska-sdp-exec-batchlet py-ska-sdp-instrumental-calibration"
module load $INST_MODULES

# Set output directory to user defined one OR to current working directory
OUTPUT_DIR=${OUTPUT_DIR:-$(builtin pwd)}

# Some extra variables
BATCHLET_CONFIG="${OUTPUT_DIR}/inst_batchlet_config.json"
INST_SUB_COMMAND="experimental"
INST_CACHE_DIR=${INST_CACHE_DIR:-$OUTPUT_DIR}
MONITOR_OUTPUT_DIR="${OUTPUT_DIR}/monitor_slurm_${SLURM_JOB_ID}"
EVERYBEAM_DATADIR=$(module show everybeam | grep CMAKE_PREFIX_PATH | sed 's=.*CMAKE_PREFIX_PATH \(.*\)/\.=\1/share/everybeam=')

# Create output directories
mkdir -p $OUTPUT_DIR $MONITOR_OUTPUT_DIR

# Generate and store batchlet's config
cat <<EOF > $BATCHLET_CONFIG
{
  "command": [
    "ska-sdp-instrumental-calibration",
    "$INST_SUB_COMMAND",
    "--config",
    "$INST_CONFIG",
    "--output",
    "$OUTPUT_DIR",
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
    "worker_scratch_directory": "$OUTPUT_DIR",
    "use_entry_node": true,
    "dask_cli_option": "--dask-scheduler"
  },
  "monitor": {
    "resources": {
      "level": 0,
      "save_dir": "$MONITOR_OUTPUT_DIR"
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
[[ $PIPELINE_EXIT_CODE -ne 0 ]] && echo -e "\ninst.sh: Removing monitoring plots because the pipelie failed with exit code $PIPELINE_EXIT_CODE.\n" 1>&2 && rm -rf $MONITOR_OUTPUT_DIR
