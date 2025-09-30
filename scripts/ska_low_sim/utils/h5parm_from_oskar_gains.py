#!/usr/bin/env python3
"""
Convert OSKAR gaintable h5 file, into DP3/LOFAR supported h5parm file
"""

import argparse
import os
from pathlib import Path
from typing import Iterable

from astropy.time import Time
import h5py
import numpy
import yaml

from datetime import datetime, timedelta


def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)


def list_stations_in_telescope_model(path):
    """
    Lists station names from a OSKAR telescope model path

    Args:
        path (str): The path to the telescope model directory

    Returns:
        list: A list of station names
    """
    directories = []
    for entry in sorted(Path(path).iterdir()):
        if (
            entry.is_dir()
            and Path(entry, "feed_angle.txt").exists()
            and Path(entry, "layout.txt").exists()
        ):
            directories.append(entry.name)
    return directories


def main():
    # Get command line arguments.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "oskar_gaintable",
        type=str,
        help="Path to oskar gaintable h5 file.",
    )
    parser.add_argument(
        "out_h5parm",
        nargs="?",
        type=str,
        default=None,
        help="(Optional) Path of output H5parm file",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    telescope_model = Path(cfg["tel_model"])
    start_time = cfg["fields"]["EoR2"]["Cal1"]["transit_time"]
    dump_time_sec = cfg["sampling_time_sec"]

    oskar_gaintable = args.oskar_gaintable

    out_h5parm = args.out_h5parm or f"{os.path.splitext(oskar_gaintable)[0]}.h5parm"
    os.makedirs(os.path.dirname(out_h5parm), exist_ok=True)

    # Read the OSKAR gain table, including the frequency axis.
    # Gains are complex, with dimensions (time, channel, antenna).
    with h5py.File(oskar_gaintable, "r") as h5file:
        gain_x = h5file["gain_xpol"][:]
        gain_y = h5file["gain_ypol"][:]
        freqs = h5file["freq (Hz)"][:]
        num_time = gain_x.shape[0]

    # Stack along a new last axis (pol),
    # resulting in shape (time, channel, antenna, pols).
    combined = numpy.stack((gain_x, gain_y), axis=-1)

    # Permute the axes to (time, antenna, channel, pols)
    combined = numpy.transpose(combined, (0, 2, 1, 3))

    # Convert complex values to amplitude and phase.
    amp = numpy.abs(combined)
    phase = numpy.angle(combined)

    # Create the polarisation axis.
    pols = ["XX", "YY"]

    # Create the antenna axis.
    ants = list_stations_in_telescope_model(telescope_model)
    ants = [f"s{i:04} ({ant_name})" for i, ant_name in enumerate(ants)]

    # Create the time axis.
    start_time = Time(start_time, scale="utc")
    t_0 = start_time.mjd * 86400.0  # Convert to MJD(UTC) seconds.
    d_t = dump_time_sec
    # TODO: Ensure that this logic is consistent with the logic in generate_gaintable.py
    times = numpy.linspace(0, num_time * d_t, num_time, endpoint=False)
    times += t_0 + d_t / 2.0

    # Write to H5parm.
    make_h5parm_soltab(
        out_h5parm, "amplitude000", "amplitude", pols, ants, times, freqs, amp
    )
    make_h5parm_soltab(out_h5parm, "phase000", "phase", pols, ants, times, freqs, phase)

    print(f"H5parm file written to {out_h5parm}")


def ndarray_of_null_terminated_bytes(names: Iterable[str]) -> numpy.ndarray:
    return numpy.asarray([s.encode() + b"\0" for s in names])


def make_h5parm_soltab(
    path: str,
    soltab_name: str,
    title: str,
    pols: list[str],
    antenna_names: list[str],
    times: numpy.ndarray,
    freqs: numpy.ndarray,
    values: numpy.ndarray,  # 4D array (time, antenna, channel, pols)
):
    with h5py.File(path, mode="a") as file:
        # The name of the solset is arbitrary.
        group = file.create_group(f"sol000/{soltab_name}")
        # NOTE: "TITLE" must be written as a fixed-length string
        # Only some specific titles are allowed:
        #     fulljones, tec, clock, scalargain, scalarphase, scalaramplitude,
        #     gain, phase, amplitude, rotationangle, rotationmeasure
        # gain, phase, amplitude correspond to diagonal solutions.
        group.attrs["TITLE"] = numpy.bytes_(title)

        # Adding null terminators is necessary, otherwise DP3 says it can't
        # find the antenna names in the H5Parm.
        group.create_dataset(
            "ant", data=ndarray_of_null_terminated_bytes(antenna_names)
        )
        group.create_dataset("pol", data=ndarray_of_null_terminated_bytes(pols))
        group.create_dataset("time", data=times)
        group.create_dataset("freq", data=freqs)

        # Values.
        # NOTE: "AXES" must be written as a fixed-length string
        axes = numpy.bytes_("time,ant,freq,pol")
        val_dataset = group.create_dataset("val", data=values)
        val_dataset.attrs["AXES"] = axes

        # Weights.
        weight_dataset = group.create_dataset(
            "weight", data=numpy.ones(values.shape, dtype=numpy.float16)
        )
        weight_dataset.attrs["AXES"] = axes


if __name__ == "__main__":
    main()
