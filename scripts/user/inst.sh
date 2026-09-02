#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=inst
#SBATCH --output=slurm-%j-%x.log

##########################################################################################

# Description
# -----------
# This script allows user to run the SKA SDP Instrumental Calibration pipeline,
# irrespective of where/how the pipeline was installed.
# This script uses 'batchlet' cli tool to manage dask cluster and monitor resources.
# The batchlet cli can be installed using the ska-sdp-exec-batchlet package.
# This script assumes that PATH and PYTHONPATH are already set appropriately.
# The script captures all stdout/stderr logs of itself and main application
# and writes them to a file path defined by variable 'stdout_log_file'.

# Usage
# -----
# The script can be executed either as a bash script (for local execution)
# or as a slurm script (in Slurm-based HPC environment)

# ENV=value [sbatch] inst.sh [--option [arg]] vis1.ms vis2.ms ...

# The measurement set (MSv2) data paths is passed as positional arguments.
# At least 1 MSv2 path must be passed.

# CLI inputs
# ----------
# (all are optional unless explictly stated as required)
#  --cmd NAME                Pipeline command. Default: ska-sdp-instrumental-calibration
#  --subcmd NAME             Pipeline subcommand. Default: run
#  --config PATH             Path to the YAML config.
#  --sky-model PATH          Path to sky model file, in SKA LSM (.csv) format
#  --sky-model-gleam PATH    Path to sky model file, in GLEAM (.dat) format
#  --cache-dir PATH          Path to the directory which INST uses
#                            to dump temporary visibilities
#  --extra-cli-args STR      Additional CLI args passed to pipeline command.
#                            Shell quoting rules apply.
#  --output-dir PATH         Output directory of the pipeline. If PATH exists, script
#                            will find next available path which is non-existent.
#                            Defaults to $PWD/output.
#  --report-dir PATH         Directory where monitoring reports are stored.
#                            Defaults to '<output-dir>/reports'
#  --temp-dir PATH           Directory where temporary files are stored.
#                            Defaults to '<output-dir>/.temp'
#  --reuse-dirs              If set, script will reuse the above output directories
#                            (even if they exist), possibly overwriting content.
#  --disable-dask-cluster    Disable batchlet-managed dask cluster creation.
#  --memory-per-worker S     Dask memory_per_worker value. Default: 16GB.
#  --threads-per-worker N    Dask threads_per_worker value. Default: 4.
#  --enable-monitor          Enable batchlet resource and log monitoring.
#  --disable-stdout-logs     Do not mirror main application stdout/stderr to terminal.

##########################################################################################

set -euo pipefail

print_help() {
  sed -n '/^##########################################################################################$/,/^##########################################################################################$/p' "$0" | sed '1d;$d;s/^# //'
}

# The following functions are duplicated from scripts/dev/_utils.sh because
# this script may run independently. Keep their implementations aligned with _utils.sh.
# log: Prints structured log line to **stdout** and **without color**
log() {
  local type="INFO"
  case "$1" in
  INFO | WARN | ERROR)
    type="$1"
    shift
    ;;
  esac
  local message="$*"
  local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  src="${src##*/}"
  local func="${FUNCNAME[1]:-main}"
  local line="${BASH_LINENO[0]:-0}"

  printf '1|%s|%s|%s#%s|%s|%s\n' \
    "$timestamp" "$src" "$func" "$line" "$type" "$message"
}
# unique_dir: Find the next available directory name
unique_dir() {
  local -n unique_dir_ref="$1"

  if [[ -e "$unique_dir_ref" ]]; then
    log WARN "Directory: ${unique_dir_ref@Q} already exists. Creating a new one."
    for ((i = 1; ; i++)); do
      if [[ ! -e "$unique_dir_ref-$i" ]]; then
        break
      fi
    done
    unique_dir_ref="$unique_dir_ref-$i"
    log "New directory: ${unique_dir_ref@Q}"
  fi
}
# log_command_paths: Log the resolved paths of the specified commands.
log_command_paths() {
  local command_name command_path

  for command_name in "$@"; do
    command_path="$(command -v "$command_name" || true)"
    if [[ -n "$command_path" ]]; then
      log "Command: ${command_name@Q} resolved to ${command_path@Q}"
    else
      log WARN "Command: ${command_name@Q} was not found in PATH"
    fi
  done
}
# join_array: Quotes each element of an array and joins them with the given delimiter
join_array() {
  if [ "$#" -lt 1 ]; then
    log ERROR "requires an array name."
    return 1
  fi

  local -n arr_ref="$1"
  local delimiter="${2:- }"
  local elem joined=""

  for elem in "${arr_ref[@]}"; do
    joined+="${elem@Q}${delimiter}"
  done

  printf '%s' "${joined%"$delimiter"}"
}

# Capture the invocation before the arg-parsing loop shifts "$@" away.
invocation_args=("$0" "$@")

cmd="ska-sdp-instrumental-calibration"
subcmd="run"
config_path=""
sky_model=""
sky_model_gleam=""
cache_dir=""
extra_cli_args=""
output_dir="$(builtin pwd)/output"
default_report_dir="reports" # Relative to output_dir.
default_temp_dir=".temp"     # Relative to output_dir.
report_dir=""
temp_dir=""
reuse_dirs=False
disable_dask_cluster=False
memory_per_worker="16GB"
threads_per_worker="4"
enable_monitor=False
disable_stdout_logs=False
ms_paths=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
  --cmd)
    cmd=${2:?Missing value for --cmd}
    shift 2
    ;;
  --subcmd)
    subcmd=${2:?Missing value for --subcmd}
    shift 2
    ;;
  --config)
    config_path=${2:?Missing value for --config}
    shift 2
    ;;
  --sky-model)
    sky_model=${2:?Missing value for --sky-model}
    shift 2
    ;;
  --sky-model-gleam)
    sky_model_gleam=${2:?Missing value for --sky-model-gleam}
    shift 2
    ;;
  --cache-dir)
    cache_dir=${2:?Missing value for --cache-dir}
    shift 2
    ;;
  --extra-cli-args)
    extra_cli_args=${2:?Missing value for --extra-cli-args}
    shift 2
    ;;
  --output-dir)
    output_dir=${2:?Missing value for --output-dir}
    shift 2
    ;;
  --report-dir)
    report_dir=${2:?Missing value for --report-dir}
    shift 2
    ;;
  --temp-dir)
    temp_dir=${2:?Missing value for --temp-dir}
    shift 2
    ;;
  --reuse-dirs)
    reuse_dirs=True
    shift
    ;;
  --disable-dask-cluster)
    disable_dask_cluster=True
    shift
    ;;
  --memory-per-worker)
    memory_per_worker=${2:?Missing value for --memory-per-worker}
    shift 2
    ;;
  --threads-per-worker)
    threads_per_worker=${2:?Missing value for --threads-per-worker}
    shift 2
    ;;
  --enable-monitor)
    enable_monitor=True
    shift
    ;;
  --disable-stdout-logs)
    disable_stdout_logs=True
    shift
    ;;
  --help | -h)
    print_help
    exit 0
    ;;
  --)
    shift
    while [[ "$#" -gt 0 ]]; do
      ms_paths+=("$1")
      shift
    done
    ;;
  -*)
    log ERROR "Unknown option: $1"
    exit 1
    ;;
  *)
    ms_paths+=("$1")
    shift
    ;;
  esac
done

if [[ "${#ms_paths[@]}" -lt 1 ]]; then
  log ERROR "At least one measurement set path must be passed as positional arg. Exiting."
  exit 1
fi

[[ $reuse_dirs == False ]] && unique_dir output_dir

report_dir=${report_dir:-"${output_dir}/${default_report_dir}"}
[[ $reuse_dirs == False ]] && unique_dir report_dir

temp_dir=${temp_dir:-"${output_dir}/${default_temp_dir}"}
[[ $reuse_dirs == False ]] && unique_dir temp_dir

mkdir -p "$output_dir" "$report_dir" "$temp_dir"

stdout_log_file="$output_dir/captured.log"
log "Captured stdout/stderr logs are stored at: ${stdout_log_file@Q}"

if [[ "$disable_stdout_logs" == True ]]; then
  log WARN "Disabling terminal output.. if required monitor ${stdout_log_file@Q} file.."
  exec >"$stdout_log_file" 2>&1
else
  exec > >(tee "$stdout_log_file") 2>&1
fi

log "Output paths are set to:
  output_dir=${output_dir@Q}
  report_dir=${report_dir@Q}
  temp_dir=${temp_dir@Q}"

script_copy_path="$temp_dir/$(basename "${BASH_SOURCE[0]}")"
cp "${BASH_SOURCE[0]}" "$script_copy_path"
log "A copy of run script is stored at: ${script_copy_path@Q}"

log "Run script invocation command: $(join_array invocation_args)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  slurm_vars_to_print=(SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURM_JOB_NUM_NODES SLURM_JOB_NODELIST)
  slurm_log="Run script executed as SLURM job, with:"
  for slurm_var in "${slurm_vars_to_print[@]}"; do
    [[ -v "$slurm_var" ]] || continue
    slurm_value="${!slurm_var}"
    slurm_log+=$'\n  '"${slurm_var}=${slurm_value@Q}"
  done
  log "$slurm_log"
fi

# set dask config
export DASK_CONFIG="${temp_dir}/dask_custom_config.yaml"

cat <<EOF >"$DASK_CONFIG"
dask:
  temporary-directory: "${temp_dir}"
distributed:
  comm:
    timeouts:
      connect: '600s'
      tcp: '900s'
  scheduler:
    worker-ttl: '15 minutes'
EOF

# Generate and store batchlet's config
batchlet_config_path="${temp_dir}/batchlet_config_inst.json"

python3 - "${ms_paths[@]}" <<EOF
import json
import shlex
import sys


batchlet_config = {}

command = [
    "$cmd",
    "$subcmd",
    "--output",
    "$output_dir",
    "--no-unique-output-subdir",
]

if config_path := "$config_path":
    command.extend(["--config", config_path])

if cache_dir := "$cache_dir":
    command.extend([
        "--set",
        "parameters.load_data.cache_directory",
        "$cache_dir",
    ])

if sky_model := "$sky_model":
    command.extend([
        "--set",
        "parameters.predict_vis.lsm_csv_path",
        sky_model,
    ])

if sky_model_gleam := "$sky_model_gleam":
    command.extend([
        "--set",
        "parameters.predict_vis.gleamfile",
        sky_model_gleam,
    ])

if extra_cli_args := "$extra_cli_args":
    command.extend(shlex.split(extra_cli_args))

mspaths = sys.argv[1:]
command.extend(mspaths)

batchlet_config["command"] = command

if not $disable_dask_cluster:
    batchlet_config["dask_params"] = {
        "threads_per_worker": $threads_per_worker,
        "memory_per_worker": "$memory_per_worker",
        "resources_per_worker": "process=$threads_per_worker",
        "worker_scratch_directory": "$temp_dir",
        "use_entry_node": True,
        "dask_cli_option": "--dask-scheduler",
        "dask_report_dir": "$report_dir",
    }

if $enable_monitor:
    batchlet_config["monitor"] = {
        "resources": {
            "level": 0,
            "save_dir": "$report_dir",
        },
        "logs": {
            "filter_plugins": [
                {
                    "name": "SKASDPFilter",
                    "kwargs": {"pipeline": "INST"},
                }
            ],
            "consumer_plugins": [
                {
                    "name": "CSVFile",
                    "kwargs": {"file_path": "$report_dir/events.csv"},
                }
            ],
        },
    }

batchlet_config["generate_reports_on_failure"] = True

batchlet_config_path = "$batchlet_config_path"

with open(batchlet_config_path, "w") as bcf:
    json.dump(batchlet_config, bcf, indent=2)
EOF

log "Batchlet's JSON config is stored at: ${batchlet_config_path@Q}"

log_command_paths "$cmd" batchlet

mapfile -t exported_env_names < <(compgen -e | sort)
log "Exported environment variables passed to subprocess: $(join_array exported_env_names ', ')"

log 'Running application via batchlet...'

echo $'\n-----------------------------------\n'

set +e
time batchlet run "$batchlet_config_path"
exit_code=$?
set -e

echo $'\n-----------------------------------\n'

if [[ "$exit_code" -eq 0 ]]; then
  log "Application finished successfully."
else
  log ERROR "Application failed with exit code: $exit_code"
fi

exit "$exit_code"
