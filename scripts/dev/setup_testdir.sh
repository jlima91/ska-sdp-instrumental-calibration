#!/usr/bin/env bash

##########################################################################################

# Description
# -----------
# Set up an isolated test environment for INST pipeline on SKA DP-HPC platform.
#
# Script allows installing extra packages via `uv pip`. These packages are installed
# without their dependencies. We reuse the same python executable supplied by
# Spack and a separate --prefix directory. This keeps pip-installed packages
# isolated from the Spack environment while ensuring they use the same Python
# version and ABI as the Spack modules.
#
# The script generates setup instructions to add the prefix's site-packages and bin
# directories to PYTHONPATH and PATH.
# The script also processes "editable packages" using their .pth files; their path
# entries are read and emitted as additional PYTHONPATH entries.

# Requirements
# ------------
# `git`, `module`, `uv`

# Configuration & Usage
# ---------------------
# 1. Copy this script into your (empty) test directory.
# 2. Edit revision and pip packages variable below as required
# 3. Execute as bash:
#    ```
#    bash setup_testdir.sh
#    ```

##########################################################################################

set -euo pipefail

# Path where the repo will be cloned
REPOROOT="$PWD/code/inst"
# Specify which commit/branch to install
REPO_REV=main
# Extra packages via uv pip
UV_PIP_PACKAGES=("-e $REPOROOT")

############################### NO NEED TO EDIT BELOW THIS ###############################

REPO_GIT_URL="https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git"
PIP_PACKAGES_PATH="$PWD/.pip_packages"
export UV_LINK_MODE=copy
export UV_PYTHON_DOWNLOADS=never

color_reset=$'\033[0m'
color_green=$'\033[0;32m'
color_cyan=$'\033[0;36m'

# Populate the referenced array with paths read from .pth files in site-packages.
read_pth_paths() {
  local -n paths=$1
  local site_packages_path=$2
  local pth_file
  local pth_entry

  paths=()
  while IFS= read -r -d '' pth_file; do
    while IFS= read -r pth_entry || [[ -n "$pth_entry" ]]; do
      [[ -z "$pth_entry" || "$pth_entry" == \#* || "$pth_entry" == import\ * ]] && continue
      if [[ "$pth_entry" == /* ]]; then
        paths+=("$pth_entry")
      else
        paths+=("${site_packages_path}/${pth_entry}")
      fi
    done <"$pth_file"
  done < <(find "$site_packages_path" -maxdepth 1 -type f -name '*.pth' -print0)
}

if [[ ! -d "$REPOROOT" ]]; then
  mkdir -p "$REPOROOT"
  git clone --filter=blob:none "$REPO_GIT_URL" "$REPOROOT"
  cd "$REPOROOT"
  git checkout "$REPO_REV"
  cd -
fi

final_message="
${color_green}Setup complete ...${color_reset}
"

[[ ! -L "./scripts" ]] && ln -s "${REPOROOT}/scripts"

source ./scripts/dev/_utils.sh

load_env_modules ska-sdp-spack python-venv
PYTHONEXE="$(command -v python)"
log Unloading modules
module purge

# (Re)Install pip packages if provided
if [[ ${#UV_PIP_PACKAGES[@]} -gt 0 ]]; then
  if ask_user_confirmation "(Re)Install packages in $PIP_PACKAGES_PATH?"; then
    set -x
    uv pip install \
      --python "$PYTHONEXE" \
      --prefix "$PIP_PACKAGES_PATH" \
      --index https://artefact.skao.int/repository/pypi-internal/simple \
      --no-deps \
      "${UV_PIP_PACKAGES[@]}"
    set +x
    log "Pip Packages installed successfully ..."
  else
    log "Skipping installation of packages."
  fi

  python_version=$("$PYTHONEXE" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  venv_site_packages="${PIP_PACKAGES_PATH}/lib/python${python_version}/site-packages"
  if [[ ! -d "$venv_site_packages" ]]; then
    log "ERROR: site-packages directory not found: ${venv_site_packages@Q}"
    exit 1
  fi

  # Handle .pth files generated due to installation of editable packages
  pth_paths=()
  read_pth_paths pth_paths "$venv_site_packages"

  pth_path_instructions=''
  [[ ${#pth_paths[@]} -gt 0 ]] && pth_path_instructions='# Editable packages'$'\n'
  for pth_path in "${pth_paths[@]}"; do
    pth_path_instructions+="module prepend-path PYTHONPATH ${pth_path@Q}"$'\n'
  done

  final_message+="
In ./scripts/dev/run.sh, please append ${color_cyan}ENV_SETUP_SCRIPT${color_reset}
section with these lines:

~~~~~~
log Pre-pending packages from ${PIP_PACKAGES_PATH@Q}
module prepend-path PYTHONPATH ${venv_site_packages@Q}
module prepend-path PATH '${PIP_PACKAGES_PATH}/bin'
${pth_path_instructions}~~~~~~
"
fi

printf '\n%s\n' ========================================
printf '%s\n' "$final_message"
printf '%s\n' ========================================
