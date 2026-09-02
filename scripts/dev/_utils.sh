#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# _utils.sh — Shared bash utilities for dev runner scripts.
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
  local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
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

  printf '%s%s%s|%s%s%s|%s%s%s|%s%s#%s%s|%s%s%s|%s\n' \
    "$C_PROTO" "1" "$reset" \
    "$C_TIME" "$timestamp" "$reset" \
    "$C_SRC" "$src" "$reset" \
    "$C_LOC" "$func" "$line" "$reset" \
    "$type_color" "$type" "$reset" \
    "$message" >&2
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
#   log_command_paths command1 command2
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# format_env_kv
#
# Formats a single KEY=VALUE pair with colour-coding, masking the value for
# potentially large keys (PATH, PYTHONPATH, LD_LIBRARY_PATH). Shared by
# format_and_print_cmd and multi_line_colored_env so both render env vars
# identically.
#
# Args:
#   $1 : key
#   $2 : value
#
# Output:
#   Prints the formatted "KEY=VALUE" string to stdout (no trailing newline).
#
# Example:
#   format_env_kv PATH "/usr/bin:/bin"
# ---------------------------------------------------------------------------
format_env_kv() {
  local key="$1"
  local value="$2"

  local reset=$'\033[0m'
  local c_key=$'\033[1;33m'    # bold yellow
  local c_eq=$'\033[0;37m'     # white
  local c_val=$'\033[0;33m'    # yellow
  local c_hidden=$'\033[1;31m' # bold red
  local mask_keys="^(PATH|PYTHONPATH|LD_LIBRARY_PATH)$"
  local formatted_value

  if [[ "$key" =~ $mask_keys ]]; then
    formatted_value="${c_hidden}[HIDDEN]${reset}"
  else
    formatted_value="${c_val}${value@Q}${reset}"
  fi

  printf '%s%s%s%s=%s%s' \
    "$c_key" "$key" "$reset" \
    "$c_eq" "$reset" \
    "$formatted_value"
}

# ---------------------------------------------------------------------------
# multi_line_colored_env
#
# Outputs the current process environment (`env`), one colour-coded, masked
# KEY=VALUE pair per line (see format_env_kv) to the stdout
#
# Example:
#   multi_line_colored_env
# ---------------------------------------------------------------------------
multi_line_colored_env() {
  local line key value formatted
  local output=""

  while IFS= read -r line; do
    key="${line%%=*}"
    value="${line#*=}"
    formatted="$(format_env_kv "$key" "$value")"
    output+="${formatted}"$'\n'
  done < <(env)

  printf '%s' "${output%$'\n'}"
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

  printf '\n%s [y/N]: ' "$message" >/dev/tty || {
    log WARN "No terminal available to prompt for confirmation; treating as 'no'."
    return 1
  }
  read -r response </dev/tty || {
    log WARN "No user input available; treating as 'no'."
    return 1
  }
  printf '\n'

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
# Quotes each element of a bash array and joins them with
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
    joined+="${elem@Q}${delimiter}"
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

  log INFO "Activating: ${venv_path@Q}"
  source "${venv_path}/bin/activate"
}

# ---------------------------------------------------------------------------
# append_env_var
#
# Appends a key=value string to an env-var array that will later be passed
# to `env -i` when constructing the final command.
# If value is not passed, it checks whether a variable with same name as key is defined,
# and if defined, considers its value.
# If the variable is not defined, returns without modifying the array.
#
# Args:
#   $1 : array_name — name of the caller's array variable (passed by reference).
#   $2 : key — a KEY string, can refer to an existing bash variable.
#   $3 : value - Optional value to assign. Can override key's existing value.
#
# Example:
#   declare -a app_env_vars=()
#   append_env_var app_env_vars HOME
#   append_env_var app_env_vars KEY value
# ---------------------------------------------------------------------------
append_env_var() {
  if [[ "$#" -lt 2 ]]; then
    log ERROR "requires an array name and KEY argument."
    return 1
  fi

  local -n env_ref="$1"
  local key="$2"
  local value

  if [[ "$#" -gt 2 ]]; then
    value="$3"
  elif [[ -v "$key" ]]; then
    value="${!key}"
  else
    return 0
  fi

  env_ref+=("$key=$value")
}

# ---------------------------------------------------------------------------
# append_cli_opt_from_var
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
#   append_cli_opt_from_var opt_cli_opt CACHE_DIR --cache-dir
#   # opt_cli_opt now contains: (--cache-dir /tmp/cache)
# ---------------------------------------------------------------------------
append_cli_opt_from_var() {
  if [ "$#" -ne 3 ]; then
    log ERROR "requires an array name, variable name, and flag."
    return 1
  fi

  local -n opt_ref="$1"
  local var_name="$2"
  local flag="$3"

  if [[ -v "$var_name" ]]; then
    opt_ref+=("$flag" "${!var_name}")
  fi
}

# ---------------------------------------------------------------------------
# format_and_print_cmd
#
# Formats a command array with ANSI colour coding and prints it to stdout.
#
# Args:
#   $1 : array_name — name of the caller's array variable (passed by reference)
#                     holding the full command to execute.
#
# Example:
#   full_cmd=(command subcmd --option1 value)
#   formatted="$(format_and_print_cmd full_cmd)"
#   log "Command to be executed:"$'\n'"$formatted"
# ---------------------------------------------------------------------------
format_and_print_cmd() {
  local -n _cmd_ref="$1"

  local C_RESET=$'\033[0m'
  local C_CMD=$'\033[1;32m'  # bold green
  local C_FLAG=$'\033[1;36m' # bold cyan
  local C_VAL=$'\033[0;37m'  # white

  local printable_cmd=""
  local arg escaped_arg key val is_first=1

  for arg in "${_cmd_ref[@]}"; do
    if [[ "$arg" == *=* && "$arg" != -* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      escaped_arg="$(format_env_kv "$key" "$val")"
    elif [[ "$arg" == -* ]]; then
      escaped_arg="${C_FLAG}${arg@Q}${C_RESET}"
    elif [[ "$is_first" -eq 1 ]]; then
      escaped_arg="${C_CMD}${arg@Q}${C_RESET}"
      is_first=0
    else
      escaped_arg="${C_VAL}${arg@Q}${C_RESET}"
    fi

    if [[ "$arg" == -* ]]; then
      printable_cmd+=$'\n  '"$escaped_arg "
    else
      printable_cmd+="$escaped_arg "
    fi
  done

  printf '%s' "$printable_cmd"
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

  if ask_user_confirmation 'Do you want to proceed?'; then
    log INFO "Executing..."
    exec "${_cmd_ref[@]}"
  else
    log WARN "Execution cancelled."
    exit 1
  fi
}
