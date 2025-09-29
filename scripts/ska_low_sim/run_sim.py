#!/usr/bin/env python3
"""
Python version of run_sim.sh
Runs OSKAR simulations with parameters from a YAML config file.
"""
import shutil

import argparse
import os
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path

RUN_OSKAR_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "./utils/run_oskar.py")


def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)


def run_command(cmd, logfile, **kwargs):
    with open(logfile, "a") as f:
        f.write(f"\nExecuting command:\n{'='*20}\n{' '.join(cmd)}\n\n")
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run OSKAR simulation using YAML config"
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Common simulation parameters
    start_freq_hz = cfg["simulation_start_frequency_hz"]
    end_freq_hz = cfg["simulation_end_frequency_hz"]
    channel_width_hz = cfg["correlated_channel_bandwidth_hz"]
    obs_length_mins = cfg["observing_time_mins"]
    dump_time_sec = cfg["sampling_time_sec"]

    # Required user parameters
    scenario = cfg["scenario"]
    oskar_sif = Path(cfg["oskar_sif"])
    tel_model = Path(cfg["tel_model"])

    # Gleam catalogue file and field radius
    gleam_file = cfg.get("gleam_file")
    field_radius_deg = cfg.get("field_radius_deg", 10.0)

    # Corruptions
    gaintable = cfg.get("gaintable")
    cable_delay = cfg.get("cable_delay")
    tec_screen = cfg.get("tec_screen")

    # Imaging parameters
    create_dirty_image = bool(cfg.get("create_dirty_image", False))
    image_size = cfg.get("image_size", 1024)
    pixel_size = cfg.get("pixel_size", "2arcsec")

    # Extra params
    run_oskar_extra_params = cfg.get("run_oskar_extra_params")

    if create_dirty_image:
        # Resolve wsclean / DP3 commands (use env vars if set)
        wsclean_cmd = os.environ.get("WSCLEAN_CMD", "wsclean")
        dp3_cmd = os.environ.get("DP3_CMD", "DP3")
        # Early check: wsclean must exist
        if not shutil.which(wsclean_cmd):
            raise Exception(
                f"wsclean command not found (looked for '{wsclean_cmd}'). Exiting."
            )
        # Early check: DP3 must exist
        if not shutil.which(dp3_cmd):
            raise Exception(f"DP3 command not found (looked for '{dp3_cmd}'). Exiting.")

    workdir = Path.cwd()
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    output_dir = workdir / f"{scenario}-{timestamp}"
    output_dir.mkdir(parents=True)

    # Copy telescope model to temp dir
    if tel_model.is_dir():
        temp_tel_model = output_dir / (tel_model.name + ".custom")
        shutil.copytree(tel_model, temp_tel_model)
    else:
        raise FileNotFoundError(f"Telescope model not found: {tel_model}")

    # Copy simulation config file as "sim.yaml"
    (output_dir / "sim.yaml").write_text(Path(args.config).read_text())

    # Symlink gleam file
    if gleam_file:
        (output_dir / "GLEAM_EGC.fits").symlink_to(Path(gleam_file).resolve())

    # Add gaintable
    if gaintable:
        (temp_tel_model / "gain_model.h5").symlink_to(Path(gaintable).resolve())

    # Add cable delay
    if cable_delay:
        (temp_tel_model / "cable_length_error.txt").symlink_to(
            Path(cable_delay).resolve()
        )

    # Build singularity command
    application = [
        "singularity",
        "exec",
        "-H",
        str(workdir),
        "--nv",
        str(oskar_sif),
    ]

    options = [
        sys.executable,
        str(RUN_OSKAR_SCRIPT_PATH),
        "--output-dir",
        str(output_dir),
        "--tel-model",
        str(temp_tel_model),
        "--obs-length-mins",
        str(obs_length_mins),
        "--dump-time-sec",
        str(dump_time_sec),
        "--start-freq-hz",
        str(start_freq_hz),
        "--end-freq-hz",
        str(end_freq_hz),
        "--channel-width-hz",
        str(channel_width_hz),
        "--field",
        "EoR2",
        "--target",
        "Cal1",
        "--scan-index",
        "0",
        "--num-scans",
        "1",
    ]

    if tec_screen:
        options += ["--tec-screen", str(Path(tec_screen).resolve())]

    if gleam_file:
        options += ["--add-gleam", "--field-radius-deg", str(field_radius_deg)]
    else:
        options += ["--no-add-gleam"]

    if run_oskar_extra_params:
        options += run_oskar_extra_params.split(" ")

    # Final command
    cmd = application + options

    logfile = output_dir / "run_sim_py.out"
    with open(logfile, "w") as f:
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Current directory: {workdir}\n")

    # Run OSKAR
    run_command(cmd, logfile, cwd=output_dir)

    # Generate dirty image
    if create_dirty_image:
        sim_ms = next(output_dir.glob("*.ms"))
        beam_corrected_ms = output_dir / f"{sim_ms.name}.beamcor.ms"
        image_name = output_dir / f"{scenario}-wsclean"

        # Apply beam using DP3
        subprocess.run(
            [
                dp3_cmd,
                f"msin={sim_ms}",
                "steps=[applybeam]",
                f"msout={beam_corrected_ms}",
            ],
            check=True,
        )

        # Run wsclean
        subprocess.run(
            [
                wsclean_cmd,
                "-size",
                str(image_size),
                str(image_size),
                "-scale",
                str(pixel_size),
                "-niter",
                "0",
                "-apply-primary-beam",
                "-name",
                str(image_name),
                str(beam_corrected_ms),
            ],
            check=True,
        )

        # Cleanup
        shutil.rmtree(beam_corrected_ms)


if __name__ == "__main__":
    main()
