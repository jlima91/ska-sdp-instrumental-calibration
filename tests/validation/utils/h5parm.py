"""
Convert an OSKAR gaintable h5 file into a DP3/LOFAR supported H5Parm file.

Copied from the converter script of scripts/ska_low_sim, so that the
validation tests do not depend on it.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Union

import h5py
import numpy
from astropy.time import Time

# Polarisations of the gains in the OSKAR gain table
POLS = ["XX", "YY"]


def list_stations_in_telescope_model(path: Union[str, Path]) -> List[str]:
    """
    List the station names of an OSKAR telescope model.

    Parameters
    ----------
    path
        The telescope model directory.

    Returns
    -------
        Names of the station directories, sorted.
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


def ndarray_of_null_terminated_bytes(names: Iterable[str]) -> numpy.ndarray:
    """
    Encode names as null terminated bytes.

    Parameters
    ----------
    names
        Names to encode.

    Returns
    -------
        Array of null terminated bytes.
    """
    return numpy.asarray([s.encode() + b"\0" for s in names])


def make_h5parm_soltab(
    path: str,
    soltab_name: str,
    title: str,
    pols: List[str],
    antenna_names: List[str],
    times: numpy.ndarray,
    freqs: numpy.ndarray,
    values: numpy.ndarray,
):
    """
    Append a solution table to an H5Parm file.

    Parameters
    ----------
    path
        H5Parm file, created if it does not exist.
    soltab_name
        Name of the solution table in the solset "sol000".
    title
        Type of the solutions (e.g. "amplitude", "phase").
    pols
        Polarisation axis.
    antenna_names
        Antenna axis.
    times
        Time axis, in MJD(UTC) seconds.
    freqs
        Frequency axis, in Hz.
    values
        Solutions, with shape (time, antenna, channel, pol).
    """
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
        group.create_dataset(
            "pol", data=ndarray_of_null_terminated_bytes(pols)
        )
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


def oskar_gaintable_to_h5parm(
    gaintable_path: str,
    h5parm_path: str,
    telescope_model: Union[str, Path],
    start_time: datetime,
    sampling_time_sec: float,
):
    """
    Convert an OSKAR gaintable into amplitude and phase solution tables of
    an H5Parm file. An existing H5Parm file is replaced.

    Parameters
    ----------
    gaintable_path
        OSKAR gaintable h5 file, with complex gains "gain_xpol" and
        "gain_ypol" of shape (time, channel, antenna), and "freq (Hz)".
    h5parm_path
        Output H5Parm file.
    telescope_model
        OSKAR telescope model, for the antenna names.
    start_time
        Start time (UTC) of the first time sample.
    sampling_time_sec
        Time between the gain solutions, in seconds.
    """
    with h5py.File(gaintable_path, "r") as h5file:
        gain_x = h5file["gain_xpol"][:]
        gain_y = h5file["gain_ypol"][:]
        freqs = h5file["freq (Hz)"][:]

    # Stack along a new last axis (pol), then permute the axes to
    # (time, antenna, channel, pols)
    gains = numpy.transpose(
        numpy.stack((gain_x, gain_y), axis=-1), (0, 2, 1, 3)
    )
    num_time = gains.shape[0]

    ants = [
        f"s{i:04} ({name})"
        for i, name in enumerate(
            list_stations_in_telescope_model(telescope_model)
        )
    ]

    # Centres of the time samples, in MJD(UTC) seconds
    t_0 = Time(start_time, scale="utc").mjd * 86400.0
    times = t_0 + (numpy.arange(num_time) + 0.5) * sampling_time_sec

    # make_h5parm_soltab appends to the file, which fails if the soltabs
    # already exist
    if os.path.exists(h5parm_path):
        os.remove(h5parm_path)
    for soltab_name, title, values in [
        ("amplitude000", "amplitude", numpy.abs(gains)),
        ("phase000", "phase", numpy.angle(gains)),
    ]:
        make_h5parm_soltab(
            h5parm_path, soltab_name, title, POLS, ants, times, freqs, values
        )
