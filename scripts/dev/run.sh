#!/usr/bin/env bash

# Description
# -----------
# This script is a dev-focused wrapper for running the SKA SDP Instrumental Calibration
# pipeline on AWS/HPC clusters. It handles:
#   1. Environment setup via module load (optionally spack load via _utils functions)
#   2. Building and confirming the final command with user
#   3. Internally launching the pipeline via scripts/user/inst.sh (wrapped in batchlet)
#
# IMPORTANT: This script must ALWAYS run as bash (not via sbatch).
# It internally uses ENTRYEXEC to call inst.sh either via sbatch (for SLURM) or
# bash (for local execution), based on your configuration.

# Configuration & Usage
# ---------------------
# 1. Set REPOROOT (if not already in environment):
#      export REPOROOT=/path/to/ska-sdp-instrumental-calibration
#
# 2. Edit configuration section below as needed
#
# 3. Execute as bash (NEVER use sbatch):
#      bash scripts/dev/run.sh
#
# The script will print the full command before executing, and you must confirm
# with 'y' to proceed. Internally, it will submit inst.sh to the HPC queue via
# sbatch (as configured in ENTRYEXEC), or run locally via bash.
# For other platforms or to disable SLURM, edit the ENTRYEXEC configuration.

set -euo pipefail

: ${REPOROOT:?is not set. Please set REPOROOT to the e2e repo root path}

source "${REPOROOT}/scripts/dev/_utils.sh"

USER_SCRIPT="${REPOROOT}/scripts/user/inst.sh"

# ========== AWS/HPC SLURM CONFIGURATION ==========
JIRA=dhr_XXX
NODES=1
PARTITION='hpc8a-96xl-ond'
MEMORY_PER_WORKER="16GB"
THREADS_PER_WORKER=4

SCENARIO="e2e-dev-run"

ENTRYEXEC=(
    sbatch
    --partition "$PARTITION"
    --nodes "$NODES"
    --exclusive
    '-J' "${JIRA}-${SCENARIO}"
)
# For local execution without sbatch:
# ENTRYEXEC=('bash')

# ========== INST PIPELINE DEPENDENCIES ==========
# Note: INST dependencies are ALWAYS loaded from metamodules or spack, never from pip.
# Module-loaded libraries do not mix well with pip site-packages.
# Loading INST module is enough to load all of its dependencies
MODULES=(
   py-ska-sdp-benchmark-monitor
   py-ska-sdp-exec-batchlet
   py-ska-sdp-instrumental-calibration
)

load_env_modules ska-sdp-spack "${MODULES[@]}"

# Override the INST package from REPOROOT
export PYTHONPATH="${REPOROOT}/src:${PYTHONPATH:-}"
export PATH="${REPOROOT}/bin:${PATH}"

# ========== PIPELINE EXECUTION CONFIGURATION ==========

COMMAND="ska-sdp-instrumental-calibration"
SUBCOMMAND="run"

INPUT_MSES="/path/to/calibrator.ms"
CONFIG="$REPOROOT/configs/calibrator_inst_run.yml"
# GLEAM_SKYMODEL=/path/to/gleamegc.dat
# SKA_SKYMODEL="/path/to/sky_model.csv"

########################################### NO NEED TO EDIT BELOW THIS #######################################

echo
log Running scenario: $'\033[0;32m'$SCENARIO$'\033[0m'
log which COMMAND: $(which $COMMAND)
echo

output_dir="$(pwd)/${SCENARIO}"

declare -a app_env_vars=()
append_env_var app_env_vars "BATCHLET_DASK_CLUSTER__DASHBOARD_ADDRESS=:30088"
append_env_var app_env_vars "HOME=$HOME"
append_env_var app_env_vars "PATH=$PATH"
append_env_var app_env_vars "EXECUTION_BLOCK_ID=eb-batch-20251203-00000"
append_env_var app_env_vars "PROCESSING_BLOCK_ID=pb-batch-20251203-00001"
append_env_var app_env_vars "PROCESSING_SCRIPT_IMAGE=oci_image"
append_env_var app_env_vars "PROCESSING_SCRIPT_NAME=e2e_processign_script"
append_env_var app_env_vars "PROCESSING_SCRIPT_VERSION=1.0.0"
[[ -v PYTHONPATH ]]        && append_env_var app_env_vars "PYTHONPATH=$PYTHONPATH"
[[ -v EVERYBEAM_DATADIR ]] && append_env_var app_env_vars "EVERYBEAM_DATADIR=$EVERYBEAM_DATADIR"

declare -a opt_cli_opt=()
append_optional_cli_opt opt_cli_opt CONFIG --config
append_optional_cli_opt opt_cli_opt SKA_SKYMODEL --sky-model
append_optional_cli_opt opt_cli_opt GLEAM_SKYMODEL --sky-model-gleam

final_full_cmd=(
    env -i "${app_env_vars[@]}"
      "${ENTRYEXEC[@]}" --
        "$USER_SCRIPT"
        --cmd "$COMMAND"
        --subcmd "$SUBCOMMAND"
        --output-dir "$output_dir"
        --enable-monitor
        --memory-per-worker "$MEMORY_PER_WORKER"
        --threads-per-worker "$THREADS_PER_WORKER"
        "${opt_cli_opt[@]}"
        -- $INPUT_MSES
)

format_and_print_cmd final_full_cmd

confirm_and_exec final_full_cmd
