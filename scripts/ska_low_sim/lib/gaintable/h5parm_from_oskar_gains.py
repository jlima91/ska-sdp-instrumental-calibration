#!/usr/bin/env python3
"""
Convert OSKAR gaintable h5 file, into DP3/LOFAR supported h5parm file
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy
from astropy.time import Time

# sim_config.py is in the parent directory (lib/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim_config import SimConfig  # noqa: E402

# Polarisations of the gains in the OSKAR gain table
POLS = ["XX", "YY"]


@dataclass
class OskarGains:
    """
    Gains of an OSKAR gain table, stacked over polarisation, with shape
    (time, antenna, channel, pol).
    """

    gains: numpy.ndarray
    freqs_hz: numpy.ndarray

    @property
    def num_time(self) -> int:
        return self.gains.shape[0]

    @classmethod
    def load(cls, path) -> "OskarGains":
        # Gains are complex, with dimensions (time, channel, antenna).
        with h5py.File(path, "r") as h5file:
            gain_x = h5file["gain_xpol"][:]
            gain_y = h5file["gain_ypol"][:]
            freqs = h5file["freq (Hz)"][:]
        # Stack along a new last axis (pol), then permute the axes to
        # (time, antenna, channel, pols)
        combined = numpy.stack((gain_x, gain_y), axis=-1)
        return cls(gains=numpy.transpose(combined, (0, 2, 1, 3)), freqs_hz=freqs)


def parse_args():
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
    return parser.parse_args()


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


def antenna_names(telescope_model) -> List[str]:
    stations = list_stations_in_telescope_model(telescope_model)
    return [f"s{i:04} ({name})" for i, name in enumerate(stations)]


def solution_times(cfg: SimConfig, num_time: int) -> numpy.ndarray:
    """
    Centres of the time samples in MJD(UTC) seconds, the same as in the MS
    simulated by simulate_visibility.py: the observation is centred on the transit time
    of the first target, and OSKAR divides it into num_time equal steps.
    """
    _, _, target = cfg.first_target()
    start_time = target.scan_start_time(cfg.observing_time_mins)
    t_0 = Time(start_time, scale="utc").mjd * 86400.0
    d_t = cfg.observing_time_mins * 60.0 / num_time
    return t_0 + (numpy.arange(num_time) + 0.5) * d_t


def ndarray_of_null_terminated_bytes(names: Iterable[str]) -> numpy.ndarray:
    return numpy.asarray([s.encode() + b"\0" for s in names])


def make_h5parm_soltab(
    path: str,
    soltab_name: str,
    title: str,
    pols: List[str],
    antenna_names: List[str],
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


def write_h5parm(path, gains: OskarGains, ants: List[str], times: numpy.ndarray):
    """Write amplitude and phase soltabs. An existing file is replaced."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # make_h5parm_soltab appends to the file, which fails if the soltabs
    # already exist
    if os.path.exists(path):
        os.remove(path)
    for soltab_name, title, values in [
        ("amplitude000", "amplitude", numpy.abs(gains.gains)),
        ("phase000", "phase", numpy.angle(gains.gains)),
    ]:
        make_h5parm_soltab(
            path, soltab_name, title, POLS, ants, times, gains.freqs_hz, values
        )


def main():
    args = parse_args()
    cfg = SimConfig.load(args.config)

    out_h5parm = (
        args.out_h5parm or f"{os.path.splitext(args.oskar_gaintable)[0]}.h5parm"
    )

    gains = OskarGains.load(args.oskar_gaintable)
    ants = antenna_names(cfg.require_simulate_visibility().tel_model)
    times = solution_times(cfg, gains.num_time)
    write_h5parm(out_h5parm, gains, ants, times)

    print(f"H5parm file written to {out_h5parm}")


if __name__ == "__main__":
    main()
