#!/usr/bin/env python3
"""
Runs OSKAR simulations with parameters from a YAML config file.

lib/run_oskar.py is run inside the OSKAR Singularity image if
simulate_visibility.oskar_sif is set, else directly with the current Python,
which must be able to import oskar.
"""

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from lib.sim_config import (
    TEL_MODEL_CORRUPTION_FILES,
    SimulateVisibilityConfig,
    SimConfig,
)

RUN_OSKAR_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "lib/run_oskar.py")

# Python modules imported by lib/run_oskar.py
RUN_OSKAR_MODULES = ("oskar", "numpy", "astropy", "yaml")


@dataclass
class ImagingTools:
    """Commands used to make the dirty image."""

    wsclean_cmd: str
    dp3_cmd: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <scenario>-<timestamp> in the current "
        "directory. If it already exists, it must be the output of a previous "
        "run, and it is replaced.",
    )
    return parser.parse_args()


def run_command(cmd, logfile, **kwargs):
    with open(logfile, "a") as f:
        f.write(f"\nExecuting command:\n{'='*20}\n{' '.join(cmd)}\n\n")
        # Written before the output of the command, which goes to the file
        # descriptor directly
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, **kwargs)


def find_imaging_tools() -> ImagingTools:
    """Resolve wsclean / DP3 commands (use env vars if set)."""
    commands = {}
    for name, env_var, default in [
        ("wsclean", "WSCLEAN_CMD", "wsclean"),
        ("DP3", "DP3_CMD", "DP3"),
    ]:
        cmd = os.environ.get(env_var, default)
        if not shutil.which(cmd):
            raise Exception(
                f"{name} command not found (looked for '{cmd}'). "
                f"Either add {name} to PATH, or set {env_var} environment "
                "variable pointing to the executable."
            )
        commands[name] = cmd
    return ImagingTools(wsclean_cmd=commands["wsclean"], dp3_cmd=commands["DP3"])


def check_oskar_available(run_cfg: SimulateVisibilityConfig):
    """Check that lib/run_oskar.py can be run, in or out of Singularity."""
    if run_cfg.oskar_sif:
        if not shutil.which("singularity"):
            raise Exception(
                "singularity command not found, but simulate_visibility.oskar_sif "
                "is set. Either add singularity to PATH, or remove "
                "simulate_visibility.oskar_sif to run OSKAR from the current "
                "Python environment."
            )
        return
    missing = [m for m in RUN_OSKAR_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise Exception(
            f"simulate_visibility.oskar_sif is not set, and {sys.executable} "
            f"cannot import {missing}. Either set simulate_visibility.oskar_sif, "
            "or install OSKAR and its Python package in this environment (see "
            "README.md)."
        )


def check_inputs_exist(run_cfg: SimulateVisibilityConfig):
    """
    Fail before creating the output directory if an input is missing. Warn if
    the telescope model contains corruptions, which lib/run_oskar.py does not
    copy.
    """
    for key, path in run_cfg.input_dirs().items():
        if not Path(path).is_dir():
            raise FileNotFoundError(f"simulate_visibility.{key} not found: {path}")
    for key, name in TEL_MODEL_CORRUPTION_FILES.items():
        if os.path.lexists(Path(run_cfg.tel_model) / name):
            print(
                f"WARNING: the telescope model contains {name}, which is not "
                f"applied. Set simulate_visibility.{key} to apply it.",
                file=sys.stderr,
            )
    for key, path in run_cfg.input_files().items():
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"simulate_visibility.{key} not found: {path}")


def check_not_deleting_inputs(
    output_dir: Path, config_path: str, run_cfg: SimulateVisibilityConfig
):
    """
    Fail if replacing output_dir would delete the current directory, the
    config file or an input (e.g. --output-dir is a parent of data/).
    """
    inputs = {**run_cfg.input_dirs(), **run_cfg.input_files()}
    protected = {"current directory": Path.cwd(), "config file": Path(config_path)}
    protected.update(
        {
            f"simulate_visibility.{key}": Path(path)
            for key, path in inputs.items()
            if path
        }
    )
    for name, path in protected.items():
        if path.resolve().is_relative_to(output_dir):
            raise RuntimeError(
                f"Not replacing {output_dir}: it contains the {name} {path}"
            )


def prepare_output_dir(
    output_dir_arg: Optional[str],
    scenario: str,
    config_path: str,
    run_cfg: SimulateVisibilityConfig,
) -> Path:
    """Create the output directory, replacing the output of a previous run."""
    if output_dir_arg:
        output_dir = Path(output_dir_arg).resolve()
        if output_dir.exists():
            if not (output_dir / "simulate_visibility_py.out").is_file():
                raise FileExistsError(
                    f"{output_dir} exists, but is not the output of a previous run"
                )
            check_not_deleting_inputs(output_dir, config_path, run_cfg)
            shutil.rmtree(output_dir)
    else:
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        output_dir = Path.cwd() / f"{scenario}-{timestamp}"
    output_dir.mkdir(parents=True)
    return output_dir


def singularity_command(
    run_cfg: SimulateVisibilityConfig,
    config_path: str,
    workdir: Path,
    output_dir: Path,
) -> List[str]:
    cmd = ["singularity", "exec", "-H", str(workdir), "--nv"]
    # Make the output directory, lib/run_oskar.py, the config file and the
    # inputs (files and directories) outside the working directory visible.
    # run_oskar.py resolves the paths (symlinks followed), so the folders of
    # the real files are bound.
    inputs = {**run_cfg.input_dirs(), **run_cfg.input_files()}
    paths = [RUN_OSKAR_SCRIPT_PATH, config_path] + [
        path for key, path in inputs.items() if path and key != "oskar_sif"
    ]
    inputs = [Path(path).resolve().parent for path in paths]
    binds = []
    for path in [output_dir] + inputs:
        if not path.is_relative_to(workdir) and path not in binds:
            binds.append(path)
    for path in binds:
        cmd += ["-B", str(path)]
    return cmd + [str(Path(run_cfg.oskar_sif).resolve())]


def run_oskar_command(
    run_cfg: SimulateVisibilityConfig, config_path: str, output_dir: Path
) -> List[str]:
    """
    Command line of lib/run_oskar.py (run from the current directory). It
    reads the simulation parameters from the config file, stages the inputs
    into the output directory, and simulates the first target.
    """
    # Inside the container, use its Python. Outside, use the current one, so
    # that the same environment (venv, PYTHONPATH) is used.
    python = "python3" if run_cfg.oskar_sif else sys.executable
    # fmt: off
    options = [
        python,
        str(RUN_OSKAR_SCRIPT_PATH),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    # fmt: on
    if run_cfg.run_oskar_extra_params:
        options += shlex.split(run_cfg.run_oskar_extra_params)
    return options


def make_dirty_image(
    output_dir: Path,
    run_cfg: SimulateVisibilityConfig,
    tools: ImagingTools,
    logfile: Path,
):
    """Apply the beam with DP3, then make a dirty image with wsclean."""
    sim_ms = next(output_dir.glob("*.ms"))
    beam_corrected_ms = output_dir / f"{sim_ms.name}.beamcor.ms"
    image_name = output_dir / "images" / "wsclean"
    image_name.parent.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            tools.dp3_cmd,
            f"msin={sim_ms}",
            "steps=[applybeam]",
            f"msout={beam_corrected_ms}",
        ],
        logfile,
    )
    # fmt: off
    run_command(
        [
            tools.wsclean_cmd,
            "-size", str(run_cfg.image_size), str(run_cfg.image_size),
            "-scale", str(run_cfg.pixel_size),
            "-niter", "0",
            "-apply-primary-beam",
            "-name", str(image_name),
            str(beam_corrected_ms),
        ],
        logfile,
    )
    # fmt: on
    shutil.rmtree(beam_corrected_ms)


def main():
    args = parse_args()
    cfg = SimConfig.load(args.config)
    run_cfg = cfg.require_simulate_visibility()

    # Early checks, before anything is created
    tools = find_imaging_tools() if run_cfg.create_dirty_image else None
    check_oskar_available(run_cfg)
    check_inputs_exist(run_cfg)

    workdir = Path.cwd()
    output_dir = prepare_output_dir(args.output_dir, cfg.scenario, args.config, run_cfg)

    cmd = []
    if run_cfg.oskar_sif:
        cmd += singularity_command(run_cfg, args.config, workdir, output_dir)
    cmd += run_oskar_command(run_cfg, args.config, output_dir)

    logfile = output_dir / "simulate_visibility_py.out"
    with open(logfile, "w") as f:
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Current directory: {workdir}\n")

    run_command(cmd, logfile)

    if tools:
        make_dirty_image(output_dir, run_cfg, tools, logfile)


if __name__ == "__main__":
    main()
