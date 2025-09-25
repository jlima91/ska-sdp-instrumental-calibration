#!/usr/bin/env python3
"""
Write H5parm file from OSKAR gain table.

Authored by:
- Maciej Serylak
- Fred Dulwich
"""

import argparse
from typing import Iterable

from astropy.time import Time
import h5py
import numpy


def main():
    # Get command line arguments.
    parser = argparse.ArgumentParser(
        prog="h5parm_from_oskar_gains",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--oskar-gains", type=str, required=True, help="Path to gain_model.h5"
    )
    parser.add_argument(
        "--start-time", type=str, required=True, help="Start time (ISO string)"
    )
    parser.add_argument(
        "--dump-time-sec", type=float, default=0.84934656, help="Dump time (s)"
    )
    parser.add_argument(
        "--h5parm", type=str, required=True, help="Path of output H5parm file"
    )
    args = parser.parse_args()

    # Read the OSKAR gain table, including the frequency axis.
    # Gains are complex, with dimensions (time, channel, antenna).
    with h5py.File(args.oskar_gains, "r") as h5file:
        gain_x = h5file["gain_xpol"][:]
        gain_y = h5file["gain_ypol"][:]
        freqs = h5file["freq (Hz)"][:]
        num_time = gain_x.shape[0]
        num_ant = gain_x.shape[2]

    # Stack along a new first axis (pol),
    # resulting in shape (pols, time, channel, antenna).
    combined = numpy.stack((gain_x, gain_y), axis=0)

    # Permute the axes to (pol, antenna, time, channel)
    combined = numpy.transpose(combined, (0, 3, 1, 2))

    # Convert complex values to amplitude and phase.
    amp = numpy.abs(combined)
    phase = numpy.angle(combined)

    # Create the polarisation axis.
    pols = ["XX", "YY"]

    # Create the antenna axis.
    ants = []
    for i in range(num_ant):
        ants.append(f"s{i:04}")

    # Create the time axis.
    start_time = Time(args.start_time, scale="utc")
    t_0 = start_time.mjd * 86400.0  # Convert to MJD(UTC) seconds.
    d_t = args.dump_time_sec
    times = numpy.linspace(0, num_time * d_t, num_time, endpoint=False)
    times += t_0 + d_t / 2.0

    # Write to H5parm.
    make_h5parm_soltab(
        args.h5parm, "amplitude000", "amplitude", pols, ants, times, freqs, amp
    )
    make_h5parm_soltab(
        args.h5parm, "phase000", "phase", pols, ants, times, freqs, phase
    )


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
    values: numpy.ndarray,  # 4D array (pol, ant, time, freq)
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
        axes = numpy.bytes_("pol,ant,time,freq")
        val_dataset = group.create_dataset("val", data=values)
        val_dataset.attrs["AXES"] = axes

        # Weights.
        weight_dataset = group.create_dataset(
            "weight", data=numpy.ones(values.shape, dtype=numpy.float16)
        )
        weight_dataset.attrs["AXES"] = axes


if __name__ == "__main__":
    main()
