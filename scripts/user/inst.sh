#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --job-name=inst
#SBATCH --output=slurm-%j-%x.out

#### HELP DOC START ####

# Description
# -----------
# This script allows user to run the ska-sdp-instrumental-calibration
# command independently of the e2e pipeline.
# This script uses 'batchlet' cli tool to manage dask cluster
# The batchlet cli can be installed using the ska-sdp-exec-batchlet package.
# This script assumes that PATH and PYTHONPATH are already set appropriately.
# The script captures all stdout/stderr logs of the main application (command + batchlet) and
# writes them to a file path defined by variable 'stdout_log_file'.

# Usage
# -----
# The script can be executed either as a bash script (for local execution)
# or as a slurm script (in Slurm-based HPC environment)

# ENV=value [sbatch] inst.sh [--option [arg]] vis1.ms vis2.ms ...

# The measurement set (MSv2) data paths is passed as positional arguments.
# At least 1 MSv2 path must be passed.

# Optional CLI inputs
# -------------------
#  --cmd NAME               Pipeline command. Default: ska-sdp-instrumental-calibration
#  --subcmd NAME            Pipeline subcommand. Default: run
#  --config PATH            Path to the YAML config provided to the run subcommand
#  --sky-model PATH         Path to sky model file, in SKA LSM (.csv) format.
#  --sky-model-gleam PATH   Path to sky model file, in GLEAM (.dat) format
#  --output-dir PATH        Output directory of the pipeline. Script ensures unique directory.
#  --cache-dir PATH         Path to the cache dir which INST uses to dump temporary visibilities.
#  --report-dir PATH        Directory where monitoring reports and dask reports are stored.
#  --temp-dir PATH          Directory where temporary files are stored, e.g. batchlet config,
#                           dask worker temporary files.
#  --extra-cli-args STR     Additional CLI args passed to pipeline command. Shell quoting rules apply.
#  --disable-dask-cluster   Disable batchlet-managed dask cluster creation.
#  --memory-per-worker S    Dask memory_per_worker value. Default: 32GB.
#  --threads-per-worker N   Dask threads_per_worker value. Default: 4.
#  --enable-monitor         Enable batchlet resource and log monitoring.
#  --disable-stdout-logs    Do not mirror main application stdout/stderr to terminal.

#### HELP DOC END ####

set -euo pipefail

print_help() {
    # Keep user-facing docs between HELP DOC markers above. If marker names change,
    # update this function too.
    sed -n '/^#### HELP DOC START ####$/,/^#### HELP DOC END ####$/p' "$0" | sed '1d;$d'
}

cmd="ska-sdp-instrumental-calibration"
subcmd="run"
config_path=""
sky_model=""
sky_model_gleam=""
output_dir="$(builtin pwd)/output"
default_report_dir="reports" # Relative to output_dir.
default_temp_dir=".temp"     # Relative to output_dir.
cache_dir=""
report_dir=""
temp_dir=""
extra_cli_args=""
disable_dask_cluster=False
memory_per_worker="32GB"
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
        --output-dir)
            output_dir=${2:?Missing value for --output-dir}
            shift 2
            ;;
        --cache-dir)
            cache_dir=${2:?Missing value for --cache-dir}
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
        --extra-cli-args)
            extra_cli_args=${2:?Missing value for --extra-cli-args}
            shift 2
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
        --help|-h)
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
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            ms_paths+=("$1")
            shift
            ;;
    esac
done

if [[ "${#ms_paths[@]}" -lt 1 ]]; then
    echo "At least one measurement set path must be passed as positional arg. Exiting."
    exit 1
fi

# Find sequentially next non-existing dir name
if [[ -e "$output_dir" ]]; then
  echo -e "Output directory: '$output_dir' already exists. Creating a new one.\n"
  for ((i=1;;i++)); do
    if [[ ! -e "$output_dir-$i" ]]; then
        break
    fi
  done
  output_dir="$output_dir-$i"
  echo -e "New output directory: '$output_dir'\n"
fi

report_dir=${report_dir:-"${output_dir}/${default_report_dir}"}
temp_dir=${temp_dir:-"${output_dir}/${default_temp_dir}"}

echo -e "Creating output paths:\n\
output_dir='$output_dir'\n\
report_dir='$report_dir'\n\
temp_dir='$temp_dir'\n"

mkdir -p "$output_dir" "$report_dir" "$temp_dir"

# set dask config
export DASK_CONFIG="${temp_dir}/dask_custom_config.yaml"

cat <<EOF > "$DASK_CONFIG"
distributed:
  comm:
    timeouts:
      connect: '600s'
      tcp: '900s'
  scheduler:
    worker-ttl: '15 minutes'
EOF

# Generate and store batchlet's config
batchlet_config_path="${temp_dir}/inst_batchlet_config.json"

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
        "resources_per_worker": "process=1",
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

print("Batchlet's JSON config is stored at: '", batchlet_config_path, "'\n")
EOF

stdout_log_file="$output_dir/captured.log"
echo -e "Captured stdout/stderr logs are stored at: '$stdout_log_file'\n"

echo -e "Running application via batchlet...\n-----------------------------------\n"

set +e
if [[ "$disable_stdout_logs" == True ]]; then
  { time batchlet run "$batchlet_config_path"; } &> "$stdout_log_file"
  exit_code=$?
else
  { time batchlet run "$batchlet_config_path"; } |& tee "$stdout_log_file"
  exit_code=${PIPESTATUS[0]}
fi
set -e

if [[ "$exit_code" -eq 0 ]]; then
    echo -e "\nApplication finished successfully."
else
    echo -e "\nApplication failed with exit code: $exit_code"
fi

exit "$exit_code"
