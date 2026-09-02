#!/usr/bin/env bash

##########################################################################################

# Description
# -----------
# This script is a dev-focused wrapper for running the SKA SDP Instrumental Calibration
# pipeline on SKA DP-HPC platform. It handles:
#   1. Environment setup via module load (optionally spack load via _utils functions)
#   2. Building and confirming the final command with user
#   3. Internally launching the pipeline via scripts/user/inst.sh
#
# IMPORTANT: This script must ALWAYS run as bash (not via sbatch).
# It internally uses ENTRYEXEC to call inst.sh either via sbatch (for SLURM) or
# bash (for local execution), based on your configuration.

# Configuration & Usage
# ---------------------
# 1. Edit configuration section below as needed
# 2. Execute as bash (NEVER use sbatch):
#    ```
#    bash scripts/dev/run.sh
#    ```
#
# The script will print the full command before executing, and you must confirm
# with 'y' to proceed. Internally, it will submit inst.sh to the HPC queue via
# sbatch (as configured in ENTRYEXEC), or run locally via bash.
# For other platforms or to disable SLURM, edit the ENTRYEXEC configuration.

##########################################################################################

set -euo pipefail

REPOROOT="$(cd -- "$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")/../.." && pwd)"

source "${REPOROOT}/scripts/dev/_utils.sh"

JIRA=dhr_XXX
NODES=1
PARTITION='hpc8a-96xl-ond'

SCENARIO=cal_inst-main

# ::: PIPELINE EXECUTION CONFIGURATION :::

COMMAND="ska-sdp-instrumental-calibration"
SUBCOMMAND="run"

INPUT_MSES=("/path/to/calibrator.ms")
CONFIG="$REPOROOT/configs/calibrator_inst_run.yml"
# GLEAM_SKYMODEL=/path/to/gleamegc.dat
# SKA_SKYMODEL="/path/to/sky_model.csv"

# ::: SCRIPT EXECUTION CONFIGURATION :::

ENTRYEXEC=(
  sbatch
  --partition "$PARTITION"
  --nodes "$NODES"
  --exclusive
  '-J' "${JIRA}-${SCENARIO}"
)
# For local execution without sbatch
# ENTRYEXEC=('bash')

USER_SCRIPT="${REPOROOT}/scripts/user/inst.sh"

# ::: ENVIRONMENT SETUP :::

# This setup runs inside the clean environment
# immediately before ENTRYEXEC/USER_SCRIPT
ENV_SETUP_SCRIPT=$(
  cat <<'EOF'
MODULES=(
  py-ska-sdp-benchmark-monitor
  py-ska-sdp-exec-batchlet
  py-ska-sdp-instrumental-calibration
)

load_env_modules ska-sdp-spack "${MODULES[@]}"

log Overriding the INST package from REPOROOT
module prepend-path PYTHONPATH "${REPOROOT}/src"
module prepend-path PATH "${REPOROOT}/scripts/bin"
EOF
)

############################### NO NEED TO EDIT BELOW THIS ###############################

log Running scenario: $'\033[0;32m'$SCENARIO$'\033[0m'

output_dir="$(pwd)/${SCENARIO}"
unique_dir output_dir

home_dir="$output_dir/.home"
# Many tools will break if home doesn't exist
mkdir -p $home_dir

declare -a app_env_vars=()
append_env_var app_env_vars BATCHLET_DASK_CLUSTER__DASHBOARD_ADDRESS ':30088'
append_env_var app_env_vars EXECUTION_BLOCK_ID eb-batch-20251203-00000
append_env_var app_env_vars PROCESSING_BLOCK_ID pb-batch-20251203-00001
append_env_var app_env_vars PROCESSING_SCRIPT_IMAGE oci_image
append_env_var app_env_vars PROCESSING_SCRIPT_NAME e2e_processign_script
append_env_var app_env_vars PROCESSING_SCRIPT_VERSION '1.0.0'
append_env_var app_env_vars HOME "$home_dir"

declare -a opt_cli_opt=()
append_cli_opt_from_var opt_cli_opt CONFIG --config
append_cli_opt_from_var opt_cli_opt CACHE_DIR --cache-dir
append_cli_opt_from_var opt_cli_opt SKA_SKYMODEL --sky-model
append_cli_opt_from_var opt_cli_opt GLEAM_SKYMODEL --sky-model-gleam

# This script runs inside the clean environment created by env -i. It restores
# the system profile and INST environment before executing the assembled command.
inner_env_script=$(
  cat <<'INNER'
set -eu

REPOROOT="$1"
ENV_SETUP_SCRIPT="$2"
COMMAND="$3"
shift 3
user_script_cmd=("$@")

source "${REPOROOT}/scripts/dev/_utils.sh"
log Raw Environment:$'\n'"$(multi_line_colored_env)"

log Sourcing /etc/profile
set +eu; source /etc/profile; set -eu

log Evaluating ENV_SETUP_SCRIPT
eval "$ENV_SETUP_SCRIPT"
log_command_paths "$COMMAND" batchlet

log Command to be executed:$'\n'"$(format_and_print_cmd user_script_cmd)"
confirm_and_exec user_script_cmd
INNER
)

clean_env_inner_script_cmd=(
  env -i "${app_env_vars[@]}"
  /bin/bash -c "$inner_env_script" --
  "$REPOROOT"
  "$ENV_SETUP_SCRIPT"
  "$COMMAND"
  "${ENTRYEXEC[@]}" --
  "$USER_SCRIPT"
  --cmd "$COMMAND"
  --subcmd "$SUBCOMMAND"
  --output-dir "$output_dir"
  --reuse-dirs
  --enable-monitor
  "${opt_cli_opt[@]}"
  -- "${INPUT_MSES[@]}"
)

log Executing subscript in a clean shell...
exec "${clean_env_inner_script_cmd[@]}"
