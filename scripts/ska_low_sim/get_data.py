#!/usr/bin/env python3
"""
Downloads the input files referenced by a YAML config file (sim.yaml).

Every input path of the "simulate_visibility" section (the telescope model
directory tel_model, and the files) is checked. Only the missing ones are downloaded, so the script can
be run again after sim.yaml is changed, e.g. to use another telescope model
or TEC screen.

Where each file is downloaded from:
- oskar_sif: OSKAR release on GitHub, with the version from the file name
  (OSKAR-<version>-Python3.sif).
- cable_delay: cable_length_error.txt of the configured telescope model.
- Any other file: <s3_prefix>/<folder>/<file name>, where <folder> depends on
  the key (see S3_FOLDERS), and <s3_prefix> is get_data.s3_prefix.

The telescope model is downloaded without its corruption files
(cable_length_error.txt, gain_model.h5), because the model must be free of
corruptions: simulate_visibility.py adds them from cable_delay and gaintable,
only if set.

A gain table written by generate_gaintable.py, and a TEC screen written by
generate_tec.py, are not downloaded.

Downloads are written to "<path>.partial" first, and renamed when complete,
so an interrupted download is resumed (or replaced) by the next run.

Needs the "aws" CLI with a session which can read the SKAO S3 buckets, and
"wget" for the OSKAR image.

Author: Team Dhruva
"""

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from lib.sim_config import TEL_MODEL_CORRUPTION_FILES, SimConfig

OSKAR_RELEASE_URL = (
    "https://github.com/OxfordSKA/OSKAR/releases/download/{version}/{name}"
)
OSKAR_SIF_PATTERN = re.compile(r"OSKAR-(?P<version>[0-9.]+)-Python3\.sif")

# Name of the cable delay file inside the telescope models in S3
CABLE_DELAY_FILE = TEL_MODEL_CORRUPTION_FILES["cable_delay"]

# S3 folder (below s3_prefix) of the files in the "simulate_visibility" section
S3_FOLDERS = {
    "gleam_file": "sky-models",
    "lotss_file": "sky-models",
    "sky_model_file": "sky-models",
    "tec_screen": "tec",
    "gaintable": "gain-tables",
}


@dataclass
class Download:
    """An input file (or directory) of sim.yaml, and where to get it."""

    key: str
    dest: Path
    source: str
    is_dir: bool = False

    @property
    def partial(self) -> Path:
        return self.dest.with_name(self.dest.name + ".partial")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which files are present and where the missing ones "
        "would be downloaded from",
    )
    return parser.parse_args()


def oskar_sif_url(oskar_sif: Path) -> str:
    match = OSKAR_SIF_PATTERN.fullmatch(oskar_sif.name)
    if not match:
        raise ValueError(
            "simulate_visibility.oskar_sif: cannot get the OSKAR version from "
            f"{oskar_sif.name!r}. Expected a name like OSKAR-2.13.0-Python3.sif"
        )
    return OSKAR_RELEASE_URL.format(version=match["version"], name=oskar_sif.name)


def plan_downloads(cfg: SimConfig) -> List[Download]:
    """All the inputs referenced by the config, with their source."""
    if cfg.simulate_visibility is None:
        return []
    prefix = cfg.get_data.s3_prefix.rstrip("/")
    tel_model = Path(cfg.simulate_visibility.tel_model)
    tel_model_source = f"{prefix}/telescope-models/{tel_model.name}"
    downloads = [Download("tel_model", tel_model, tel_model_source + "/", is_dir=True)]

    generated_gaintable = (
        Path(cfg.generate_gaintable.output_gaintable).resolve()
        if cfg.generate_gaintable
        else None
    )
    generated_tec_screen = (
        Path(cfg.generate_tec.output_tec_screen).resolve()
        if cfg.generate_tec
        else None
    )
    for key, path in cfg.simulate_visibility.input_files().items():
        if not path:
            continue
        path = Path(path)
        if key == "oskar_sif":
            source = oskar_sif_url(path)
        elif key == "cable_delay":
            source = f"{tel_model_source}/{CABLE_DELAY_FILE}"
        elif key == "gaintable" and path.resolve() == generated_gaintable:
            print(f"[skip]    {key}: {path} (run generate_gaintable.py to create it)")
            continue
        elif key == "tec_screen" and path.resolve() == generated_tec_screen:
            print(f"[skip]    {key}: {path} (run generate_tec.py to create it)")
            continue
        else:
            source = f"{prefix}/{S3_FOLDERS[key]}/{path.name}"
        downloads.append(Download(key, path, source))
    return downloads


def run(cmd: List[str]):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch(download: Download):
    """Download into "<dest>.partial", then rename to dest."""
    dest, partial = download.dest, download.partial
    dest.parent.mkdir(parents=True, exist_ok=True)
    if download.source.startswith("https://"):
        run(["wget", "--continue", "-O", str(partial), download.source])
    elif download.is_dir:
        excludes = []
        for name in TEL_MODEL_CORRUPTION_FILES.values():
            excludes += ["--exclude", name]
        run(["aws", "s3", "sync", download.source, str(partial)] + excludes)
        # "aws s3 sync" succeeds when there is nothing at the source
        if not partial.is_dir() or not any(partial.iterdir()):
            raise RuntimeError(f"Nothing found at {download.source}")
    else:
        run(["aws", "s3", "cp", download.source, str(partial)])
    partial.rename(dest)


def check_tools(downloads: List[Download]):
    """Fail before downloading anything if a command is not available."""
    needed = {"wget" if d.source.startswith("https://") else "aws" for d in downloads}
    missing = sorted(tool for tool in needed if not shutil.which(tool))
    if missing:
        raise RuntimeError(f"Commands not found: {missing}")


def main():
    args = parse_args()
    cfg = SimConfig.load(args.config)

    missing = []
    for download in plan_downloads(cfg):
        if download.dest.exists():
            print(f"[present] {download.key}: {download.dest}")
        else:
            print(f"[missing] {download.key}: {download.dest} <- {download.source}")
            missing.append(download)

    if args.dry_run or not missing:
        return
    check_tools(missing)

    failed: List[Download] = []
    for download in missing:
        try:
            fetch(download)
        except (subprocess.CalledProcessError, RuntimeError) as err:
            print(f"Failed to download {download.key}: {err}", file=sys.stderr)
            failed.append(download)

    if failed:
        print("\nNot downloaded:", file=sys.stderr)
        for download in failed:
            print(f"  {download.key}: {download.source}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
