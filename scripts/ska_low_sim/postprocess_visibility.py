#!/usr/bin/env python3
"""
Post-processing of an MS simulated by simulate_visibility.py: writes the
field name, scan intent and scan number of the target into the MS.

OSKAR writes an empty FIELD NAME, OBSERVE_TARGET#ON_SOURCE as the STATE
OBS_MODE, and 0 as the SCAN_NUMBER. This script sets them from the target in
the "fields" section of the YAML config file (sim.yaml):
- FIELD NAME: field_id of the target (default: the target name).
- STATE OBS_MODE: scan_intent of the target (default: not changed).
- SCAN_NUMBER: the scan number in the MS name, visibility.scan-<number>.ms
  (scan_id_start of the target + scan index), or --scan-number.

The target is given with --field and --target (default: the first target of
the first field, which is the one simulated by simulate_visibility.py).

This script needs python-casacore, which is not in the OSKAR Singularity
image, so it is run on the host after simulate_visibility.py.

Usage:
python postprocess_visibility.py sim.yaml <output-dir>/visibility.scan-300.ms
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from casacore.tables import table

from lib.sim_config import MS_NAME_PATTERN, SimConfig, Target


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("ms_path", type=str, help="MS to update, in place")
    parser.add_argument(
        "--field",
        default=None,
        help="Field name in the config file. Given with --target. Default: the "
        "field of the first target",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target name in the field. Given with --field. Default: the first "
        "target of the first field",
    )
    parser.add_argument(
        "--scan-number",
        type=int,
        default=None,
        help="Scan number. Default: the number in the MS name "
        "(visibility.scan-<number>.ms)",
    )
    return parser.parse_args()


def select_target(
    cfg: SimConfig, field_name: Optional[str], target_name: Optional[str]
) -> tuple[str, Target]:
    """
    Target given by its field and target names, or the first target.

    Parameters
    ----------
    cfg
        The config file.
    field_name
        Field name, given with target_name.
    target_name
        Target name in the field, given with field_name.

    Returns
    -------
        The target name and the target.
    """
    if field_name and target_name:
        return target_name, cfg.get_target(field_name, target_name)
    if field_name or target_name:
        raise ValueError("--field and --target must be given together")
    _, target_name, target = cfg.first_target()
    return target_name, target


def scan_number_from_ms_name(ms_path: Path, target: Target) -> int:
    """
    Scan number in the name of an MS written by lib/run_oskar.py.

    Parameters
    ----------
    ms_path
        The MS, named visibility.scan-<number>.ms.
    target
        The simulated target. The scan number must be at least its
        scan_id_start.

    Returns
    -------
        The scan number.
    """
    match = MS_NAME_PATTERN.fullmatch(ms_path.name)
    if not match:
        raise ValueError(
            f"Cannot get the scan number from the MS name {ms_path.name!r} "
            "(expected visibility.scan-<number>.ms). Give it with --scan-number"
        )
    scan_number = int(match["scan_number"])
    if scan_number < target.scan_id_start:
        raise ValueError(
            f"Scan number {scan_number} of {ms_path.name} is smaller than the "
            f"scan_id_start ({target.scan_id_start}) of the target. Select "
            "the target of this MS with --field and --target"
        )
    return scan_number


def add_field_id_and_scan_intent_to_source(
    ms_path: Path,
    field_id: Optional[str],
    scan_intent: Optional[str],
    scan_number: Optional[int] = None,
):
    """
    Write the field name, scan intent and scan number into an MS. Values
    which are None are not changed.

    Parameters
    ----------
    ms_path
        The MS, updated in place.
    field_id
        NAME of all the rows of the FIELD table.
    scan_intent
        OBS_MODE of all the rows of the STATE table.
    scan_number
        SCAN_NUMBER of all the rows of the main table.
    """
    if scan_intent is not None:
        with table(str(ms_path / "STATE"), readonly=False, ack=False) as ms:
            ms.putcol("OBS_MODE", [scan_intent] * ms.nrows())

    if field_id is not None:
        with table(str(ms_path / "FIELD"), readonly=False, ack=False) as ms:
            ms.putcol("NAME", [field_id] * ms.nrows())

    if scan_number is not None:
        with table(str(ms_path), readonly=False, ack=False) as ms:
            ms.putcol("SCAN_NUMBER", np.full(ms.nrows(), scan_number, dtype=np.int32))


def main():
    args = parse_args()
    cfg = SimConfig.load(args.config)
    ms_path = Path(args.ms_path)
    if not ms_path.is_dir():
        raise FileNotFoundError(f"MS not found: {ms_path}")

    target_name, target = select_target(cfg, args.field, args.target)
    field_id = target.field_id or target_name
    scan_number = (
        args.scan_number
        if args.scan_number is not None
        else scan_number_from_ms_name(ms_path, target)
    )

    print(f"Updating {ms_path} for target {target_name}:")
    print(f"    FIELD NAME     = {field_id}")
    print(f"    STATE OBS_MODE = {target.scan_intent or '(not changed)'}")
    print(f"    SCAN_NUMBER    = {scan_number}")
    add_field_id_and_scan_intent_to_source(
        ms_path, field_id, target.scan_intent, scan_number
    )


if __name__ == "__main__":
    main()
