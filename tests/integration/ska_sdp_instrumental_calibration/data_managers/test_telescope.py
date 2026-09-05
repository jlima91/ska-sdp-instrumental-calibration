# pylint:disable=c-extension-no-member
import tarfile
import tempfile
from pathlib import Path

import everybeam as eb
import numpy as np
from astropy.coordinates import SkyCoord

from ska_sdp_instrumental_calibration.data_managers.beams import (
    convert_time_to_solution_time,
    radec_to_xyz,
)
from ska_sdp_instrumental_calibration.data_managers.telescope import Telescope

TEST_ROOT = Path(__file__).resolve().parent / "../../../"
OSKAR_TAR = TEST_ROOT / "e2e/resources/test.ms.tgz"


def get_beam_resp(telescope, nstations, frequency, solution_time, delay_dir):
    beams = np.empty(
        (
            1,
            nstations,
            frequency.size,
            2,
            2,
        ),
        dtype=np.complex128,
    )
    for stn in range(nstations):
        for chan, freq in enumerate(frequency):
            beams[0, stn, chan, :, :] = telescope.station_response(
                solution_time,
                stn,
                freq,
                delay_dir,
                delay_dir,
            )
    return beams


def test_should_validate_station_response_for_telescope():
    frequencies = np.array(
        [
            1.0005e5,
            1.0010e5,
            1.0015e5,
            1.0020e5,
        ]
    )

    time = 4453655635.070823
    direction = radec_to_xyz(
        SkyCoord(ra=197.914612, dec=-22.277973, unit="deg", frame="icrs"),
        convert_time_to_solution_time(time),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(OSKAR_TAR, "r:*") as tar:
            tar.extractall(path=temp_dir)
            eb_ms_name = tar.getnames()[0].split("/")[0]
            eb_ms = (temp_dir / Path(eb_ms_name)).as_posix()
        created_telescope = Telescope(
            eb_ms, element_response_model="oskar_dipole_cos"
        )

        assert created_telescope.type is eb.OSKAR

        nstations = created_telescope._nstations

        beam_create_telescope = created_telescope.station_response(
            time, frequencies, direction, direction
        )

        loaded_telescope = eb.load_telescope(
            eb_ms, element_response_model="oskar_dipole_cos"
        )

        beam_load_telescope = get_beam_resp(
            loaded_telescope, nstations, frequencies, time, direction
        )

        np.testing.assert_allclose(beam_load_telescope, beam_create_telescope)


def test_should_scale_station_response_for_telescope():
    frequencies = np.array(
        [
            1.0005e5,
            1.0010e5,
            1.0015e5,
            1.0020e5,
        ]
    )

    time = 4453655635.070823
    direction = radec_to_xyz(
        SkyCoord(ra=197.914612, dec=-22.277973, unit="deg", frame="icrs"),
        convert_time_to_solution_time(time),
    )

    scale = (
        np.ones(
            (frequencies.size,),
            dtype=frequencies.dtype,
        )
        / 2.0
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(OSKAR_TAR, "r:*") as tar:
            tar.extractall(path=temp_dir)
            eb_ms_name = tar.getnames()[0].split("/")[0]
            eb_ms = (temp_dir / Path(eb_ms_name)).as_posix()
        created_telescope = Telescope(
            eb_ms, element_response_model="oskar_dipole_cos"
        )

        assert created_telescope.type is eb.OSKAR

        beam_create_telescope = created_telescope.station_response(
            time, frequencies, direction, direction
        )

        beam_create_telescope_scaled = created_telescope.station_response(
            time,
            frequencies,
            direction,
            direction,
            scale,
        )

        np.testing.assert_allclose(
            beam_create_telescope / 2.0,
            beam_create_telescope_scaled,
        )


def test_should_validate_station_response_for_single_station():
    frequency = 1.0005e5

    time = 4453655635.070823
    direction = radec_to_xyz(
        SkyCoord(ra=197.914612, dec=-22.277973, unit="deg", frame="icrs"),
        convert_time_to_solution_time(time),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(OSKAR_TAR, "r:*") as tar:
            tar.extractall(path=temp_dir)
            eb_ms_name = tar.getnames()[0].split("/")[0]
            eb_ms = (temp_dir / Path(eb_ms_name)).as_posix()
        created_telescope = Telescope(
            eb_ms, element_response_model="oskar_dipole_cos"
        )

        loaded_telescope = eb.load_telescope(
            eb_ms, element_response_model="oskar_dipole_cos"
        )

        np.testing.assert_allclose(
            loaded_telescope.station_response(
                time, 0, frequency, direction, direction
            ),
            created_telescope.station_response(
                time,
                frequency,
                direction,
                direction,
                station_idx=0,
            ).squeeze(axis=(0, 1, 2)),
        )
