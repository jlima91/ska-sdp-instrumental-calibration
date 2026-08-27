#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#
# _utils.sh — Shared bash utilities for dev runner scripts.
#
# Usage:
#   REPOROOT="/path/to/ska-sdp-instrumental-calibration"
#   source "${REPOROOT}/scripts/dev/_utils.sh"
#
# The runner script is then responsible for:
#   1. Defining REPOROOT and any environment-specific path variables.
#   2. Calling the environment setup helpers (load_spack_modules,
#      load_env_modules, activate_python_environment) as needed for the
#      target platform.
#   3. Building the app_env_vars and opt_cli_opt arrays using
#      append_env_var / append_optional_cli_opt.
#   4. Calling format_and_print_cmd and confirm_and_exec to preview
#      and launch the final command.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# log
#
# Prints a colour-coded, structured log line to stderr in the format:
#   1|{timestamp}|{bash_source}|{function_name}#{line_number}|TYPE|message
#
# Args:
#   $1 : type (optional) — one of INFO, WARN, ERROR. Defaults to INFO.
#   $@ : message — the remaining arguments are joined to form the message.
#
# Example:
#   log "Starting up"
#   log WARN "Cache dir not set"
#   log ERROR "Failed to activate venv"
# ---------------------------------------------------------------------------
log() {
  local type="INFO"
  case "$1" in
  INFO | WARN | ERROR)
    type="$1"
    shift
    ;;
  esac
  local message="$*"
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  src="${src##*/}"
  local func="${FUNCNAME[1]:-main}"
  local line="${BASH_LINENO[0]:-0}"

  local reset=$'\033[0m'
  local C_PROTO=$'\033[0;90m' # grey
  local C_TIME=$'\033[0;36m'  # cyan
  local C_SRC=$'\033[0;94m'   # bright blue
  local C_LOC=$'\033[0;35m'   # magenta
  local type_color
  case "$type" in
  INFO) type_color=$'\033[0;32m' ;;  # green
  WARN) type_color=$'\033[0;33m' ;;  # yellow
  ERROR) type_color=$'\033[0;31m' ;; # red
  esac

  echo -e "${C_PROTO}1${reset}|${C_TIME}${timestamp}${reset}|${C_SRC}${src}${reset}|${C_LOC}${func}#${line}${reset}|${type_color}${type}${reset}|${message}" >&2
}

# ---------------------------------------------------------------------------
# unique_dir
#
# Finds the next available directory name by appending a numeric suffix when
# the requested directory already exists.
#
# Args:
#   $1 : variable_name — name of the caller variable containing the directory
#                        path, passed by reference.
#
# Side effects:
#   - Updates the caller's directory variable with the unique path.
#
# Example:
#   output_dir="/path/to/output"
#   unique_dir output_dir
# ---------------------------------------------------------------------------
unique_dir() {
  local -n unique_dir_ref="$1"

  if [[ -e "$unique_dir_ref" ]]; then
    log WARN "Directory: '$unique_dir_ref' already exists. Creating a new one."
    for ((i = 1; ; i++)); do
      if [[ ! -e "$unique_dir_ref-$i" ]]; then
        break
      fi
    done
    unique_dir_ref="$unique_dir_ref-$i"
    log "New directory: '$unique_dir_ref'"
  fi
}

# ---------------------------------------------------------------------------
# log_command_paths
#
# Logs the resolved executable path for each supplied command. This records
# which executable is selected by the current PATH.
#
# Args:
#   $@ : command names to resolve.
#
# Example:
#   log_command_paths ska-sdp-instrumental-calibration batchlet
# ---------------------------------------------------------------------------
log_command_paths() {
  local command_name command_path

  for command_name in "$@"; do
    command_path="$(command -v "$command_name" || true)"
    if [[ -n "$command_path" ]]; then
      log "Command: '$command_name' resolved to '$command_path'"
    else
      log WARN "Command: '$command_name' was not found in PATH"
    fi
  done
}

# ---------------------------------------------------------------------------
# ask_user_confirmation
#
# Prompts the user for a yes/no response (y/N) on /dev/tty.
# Returns 0 for yes and 1 for no or missing input.
#
# Args:
#   $1 : message — prompt text without the [y/N] suffix.
#
# Example:
#   if ask_user_confirmation "Create venv?"; then
#     echo "yes"
#   fi
# ---------------------------------------------------------------------------
ask_user_confirmation() {
  if [ "$#" -ne 1 ]; then
    log ERROR "requires one prompt message argument."
    return 1
  fi

  local message="$1"
  local response

  read -r -p "${message} [y/N] " response </dev/tty || {
    log WARN "No user input available; treating as 'no'."
    return 1
  }

  case "$response" in
  [yY][eE][sS] | [yY])
    return 0
    ;;
  *)
    return 1
    ;;
  esac
}

# ---------------------------------------------------------------------------
# join_array
#
# Quotes each element of a bash array (via printf '%q') and joins them with
# the given delimiter, trimming any trailing delimiter. Preserves elements
# containing spaces/special characters, unlike "${arr[*]}" expansion.
#
# Args:
#   $1 : array_name — name of the caller's array variable (passed by reference).
#   $2 : delimiter (optional) — string to join elements with. Defaults to ' '.
#
# Output:
#   Prints the joined, quoted string to stdout.
#
# Example:
#   modules=(py-foo py-bar)
#   joined="$(join_array modules)"
#   joined="$(join_array modules ', ')"
# ---------------------------------------------------------------------------
join_array() {
  if [ "$#" -lt 1 ]; then
    log ERROR "requires an array name."
    return 1
  fi

  local -n arr_ref="$1"
  local delimiter="${2:- }"
  local elem joined=""

  for elem in "${arr_ref[@]}"; do
    joined+="$(printf '%q' "$elem")${delimiter}"
  done

  printf '%s' "${joined%"$delimiter"}"
}

# ---------------------------------------------------------------------------
# load_spack_modules
#
# Unloads any active Spack packages, activates the given Spack environment,
# and optionally loads a list of named packages.
#
# Args:
#   $1 : spack_root      — absolute path to the Spack installation root
#                          (the directory containing bin/spack).
#   $2 : spack_env_path  — absolute path to the Spack environment to activate.
#   $@ : modules         — one or more Spack package specs to load after
#                          environment activation.
#
# Example:
#   load_spack_modules /opt/spack /path/to/env \
#       py-ska-sdp-exec-batchlet py-ska-sdp-instrumental-calibration
# ---------------------------------------------------------------------------
load_spack_modules() {
  if [ "$#" -lt 3 ]; then
    log ERROR "requires a spack_root, spack_env_path, and at least 1 module."
    return 1
  fi

  local spack_root="$1"
  local spack_env_path="$2"
  shift 2
  local modules=("$@")

  eval $("${spack_root}/bin/spack" unload --sh --all)
  eval $("${spack_root}/bin/spack" env activate --sh "$spack_env_path")

  log INFO "Spack loading: $(join_array modules)"
  eval $("${spack_root}/bin/spack" load --sh "${modules[@]}")
}

# ---------------------------------------------------------------------------
# load_env_modules
#
# Purges all loaded environment modules, then loads the specified modules
# using the system `module` command. Exits the calling script on failure.
#
# Args:
#   $@ : modules — one or more module names to load (required).
#
# Example:
#   load_env_modules ska-sdp-spack py-ska-sdp-instrumental-calibration
# ---------------------------------------------------------------------------
load_env_modules() {
  if [ "$#" -eq 0 ]; then
    log ERROR "requires at least one module."
    return 1
  fi

  local modules=("$@")

  module purge
  log INFO "Loading modules: $(join_array modules)"
  module -s load "${modules[@]}" || {
    log ERROR "module load failed!"
    exit 1
  }
}

# ---------------------------------------------------------------------------
# activate_python_environment
#
# Sources the activate script for a given Python virtual environment and,
#
# Args:
#   $1 : venv_path — absolute path to the virtual environment root.
#
# Side effects:
#   - Activates the venv (modifies PATH and VIRTUAL_ENV in the caller).
#
# Example:
#   activate_python_environment "${REPOROOT}/.venv"
# ---------------------------------------------------------------------------
activate_python_environment() {
  if [ "$#" -lt 1 ]; then
    log ERROR "requires a venv path."
    return 1
  fi

  local venv_path="$1"

  log INFO "Activating: ${venv_path}"
  source "${venv_path}/bin/activate"
}

# ---------------------------------------------------------------------------
# append_env_var
#
# Appends a KEY=VALUE string to an env-var array that will later be passed
# to `env -i` when constructing the final command.
#
# Args:
#   $1 : array_name — name of the caller's array variable (passed by reference).
#   $2 : assignment — a KEY=VALUE string.
#
# Example:
#   declare -a app_env_vars=()
#   append_env_var app_env_vars "HOME=$HOME"
# ---------------------------------------------------------------------------
append_env_var() {
  if [ "$#" -ne 2 ]; then
    log ERROR "requires an array name and KEY=VALUE argument."
    return 1
  fi

  local -n env_ref="$1"
  env_ref+=("$2")
}

# ---------------------------------------------------------------------------
# append_optional_cli_opt
#
# Appends a flag and its value to a CLI-options array only when the named
# variable is set (even if empty). Does nothing if the variable is unset.
#
# Args:
#   $1 : array_name  — name of the caller's array variable (passed by reference).
#   $2 : var_name    — name of the variable whose value supplies the flag argument.
#   $3 : flag        — the CLI flag string (e.g. --cache-dir).
#
# Example:
#   declare -a opt_cli_opt=()
#   CACHE_DIR="/tmp/cache"
#   append_optional_cli_opt opt_cli_opt CACHE_DIR --cache-dir
#   # opt_cli_opt now contains: (--cache-dir /tmp/cache)
# ---------------------------------------------------------------------------
append_optional_cli_opt() {
  if [ "$#" -ne 3 ]; then
    log ERROR "requires an array name, variable name, and flag."
    return 1
  fi

  local -n opt_ref="$1"
  local var_name="$2"
  local flag="$3"

  if [ -n "${!var_name+x}" ]; then
    opt_ref+=("$flag" "${!var_name}")
  fi
}

# ---------------------------------------------------------------------------
# format_and_print_cmd
#
# Pretty-prints a command array with ANSI colour coding:
#
# Args:
#   $1 : array_name — name of the caller's array variable (passed by reference)
#                     holding the full command to execute.
#
# Example:
#   format_and_print_cmd final_full_cmd
# ---------------------------------------------------------------------------
format_and_print_cmd() {
  local -n _cmd_ref="$1"

  local C_RESET=$'\033[0m'
  local C_CMD=$'\033[1;32m'
  local C_FLAG=$'\033[1;36m'
  local C_KEY=$'\033[1;33m'
  local C_EQ=$'\033[0;37m'
  local C_ENV_VAL=$'\033[0;33m'
  local C_HIDDEN=$'\033[1;31m'
  local C_VAL=$'\033[0;37m'

  local mask_keys="^(PATH|PYTHONPATH|LD_LIBRARY_PATH)="
  local printable_cmd=""
  local arg escaped_arg key val is_first=1

  for arg in "${_cmd_ref[@]}"; do
    if [[ "$arg" =~ $mask_keys ]]; then
      key="${arg%%=*}"
      escaped_arg="${C_KEY}${key}${C_EQ}=${C_HIDDEN}[HIDDEN]${C_RESET}"
    elif [[ "$arg" == *=* && "$arg" != -* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      local escaped_val
      escaped_val=$(printf '%q' "$val")
      escaped_arg="${C_KEY}${key}${C_EQ}=${C_ENV_VAL}${escaped_val}${C_RESET}"
    elif [[ "$arg" == -* ]]; then
      escaped_arg="${C_FLAG}$(printf '%q' "$arg")${C_RESET}"
    elif [[ "$is_first" -eq 1 ]]; then
      escaped_arg="${C_CMD}$(printf '%q' "$arg")${C_RESET}"
      is_first=0
    else
      escaped_arg="${C_VAL}$(printf '%q' "$arg")${C_RESET}"
    fi

    if [[ "$arg" == -* ]]; then
      printable_cmd+=$'\n  '"$escaped_arg "
    else
      printable_cmd+="$escaped_arg "
    fi
  done

  echo -e "\033[1;35m==========================================\033[0m"
  echo -e "\033[1mCommand to be executed:\033[0m"
  echo "$printable_cmd"
  echo -e "\033[1;35m==========================================\033[0m"
  echo
}

# ---------------------------------------------------------------------------
# confirm_and_exec
#
# Prompts the user for confirmation (y/N) on /dev/tty and, on approval,
# replaces the current process with the given command via `exec`.
# The script exits with code 1 if the user declines.
#
# Args:
#   $1 : array_name — name of the caller's array variable (passed by reference)
#                     holding the full command to execute.
#
# Example:
#   confirm_and_exec final_full_cmd
# ---------------------------------------------------------------------------
confirm_and_exec() {
  local -n _cmd_ref="$1"

  if ask_user_confirmation "Do you want to proceed?"; then
    log INFO "Executing..."
    exec "${_cmd_ref[@]}"
  else
    log WARN "Execution cancelled."
    exit 1
  fi
}
