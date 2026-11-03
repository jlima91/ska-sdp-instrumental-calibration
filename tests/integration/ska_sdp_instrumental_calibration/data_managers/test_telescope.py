# pylint:disable=c-extension-no-member
import pickle
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import everybeam as eb
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from casacore.tables import table

from ska_sdp_instrumental_calibration.data_managers.beams import (
    convert_time_to_solution_time,
    radec_to_xyz,
)
from ska_sdp_instrumental_calibration.data_managers.telescope import Telescope

TEST_ROOT = Path(__file__).resolve().parent / "../../../"
OSKAR_TAR = TEST_ROOT / "e2e/resources/test.ms.tgz"
ELEMENT_RESPONSE_MODEL = "oskar_dipole_cos"
TIME = 4453655635.070823
OFF_AXIS_SOURCE = SkyCoord(
    ra=199.914612, dec=-23.777973, unit="deg", frame="icrs"
)


@pytest.fixture(scope="module")
def eb_ms(tmp_path_factory):
    temp_dir: Path = tmp_path_factory.mktemp("eb_ms")
    with tarfile.open(OSKAR_TAR, "r:*") as tar:
        tar.extractall(path=temp_dir)
        eb_ms_name = tar.getnames()[0].split("/")[0]
    return (temp_dir / eb_ms_name).as_posix()


@pytest.fixture
def telescope(eb_ms):
    return Telescope(eb_ms, element_response_model=ELEMENT_RESPONSE_MODEL)


@pytest.fixture(scope="module")
def direction():
    return radec_to_xyz(
        SkyCoord(ra=197.914612, dec=-22.277973, unit="deg", frame="icrs"),
        convert_time_to_solution_time(TIME),
    )


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


def test_should_validate_station_response_for_telescope(
    eb_ms, telescope, direction
):
    frequencies = np.array(
        [
            1.0005e5,
            1.0010e5,
            1.0015e5,
            1.0020e5,
        ]
    )

    assert telescope.type is eb.OSKAR

    nstations = telescope._nstations

    beam_create_telescope = telescope.station_response(
        TIME, frequencies, direction, direction
    )

    loaded_telescope = eb.load_telescope(
        eb_ms, element_response_model=ELEMENT_RESPONSE_MODEL
    )

    beam_load_telescope = get_beam_resp(
        loaded_telescope, nstations, frequencies, TIME, direction
    )

    np.testing.assert_allclose(beam_load_telescope, beam_create_telescope)


def test_should_match_loaded_telescope_for_off_axis_direction(
    eb_ms, telescope, direction
):
    """
    With the source direction equal to the beam pointing direction, all
    element phases are zero and the station layout cancels out. Use an
    off-axis source so that the element offsets and coordinate axes of
    each station influence the response.
    """
    frequencies = np.array([1.0e8, 1.5e8, 2.0e8])
    off_axis = radec_to_xyz(
        OFF_AXIS_SOURCE, convert_time_to_solution_time(TIME)
    )

    loaded_telescope = eb.load_telescope(
        eb_ms, element_response_model=ELEMENT_RESPONSE_MODEL
    )
    nstations = telescope._nstations
    expected = np.array(
        [
            [
                loaded_telescope.station_response(
                    TIME, stn, freq, off_axis, direction
                )
                for freq in frequencies
            ]
            for stn in range(nstations)
        ]
    )

    actual = telescope.station_response(
        TIME, frequencies, off_axis, direction
    ).squeeze(axis=0)

    np.testing.assert_allclose(actual, expected)


def test_should_disable_flagged_elements(eb_ms, direction, tmp_path):
    flagged_ms = (tmp_path / "flagged.ms").as_posix()
    shutil.copytree(eb_ms, flagged_ms)
    with table(f"{flagged_ms}/PHASED_ARRAY", readonly=False, ack=False) as tb:
        element_flag = tb.getcol("ELEMENT_FLAG")
        # Flag the X dipole of the first half of the elements of station 0,
        # and the Y dipole of the last element of station 1
        nelements = element_flag.shape[2]
        element_flag[0, 0, : nelements // 2] = True
        element_flag[1, 1, -1] = True
        tb.putcol("ELEMENT_FLAG", element_flag)

    frequencies = np.array([1.0e8, 1.5e8])
    off_axis = radec_to_xyz(
        OFF_AXIS_SOURCE, convert_time_to_solution_time(TIME)
    )
    loaded_telescope = eb.load_telescope(
        flagged_ms, element_response_model=ELEMENT_RESPONSE_MODEL
    )
    expected = np.array(
        [
            [
                loaded_telescope.station_response(
                    TIME, stn, freq, off_axis, direction
                )
                for freq in frequencies
            ]
            for stn in range(2)
        ]
    )

    actual = Telescope(
        flagged_ms, element_response_model=ELEMENT_RESPONSE_MODEL
    ).station_response(TIME, frequencies, off_axis, direction)[0, :2]
    unflagged = Telescope(
        eb_ms, element_response_model=ELEMENT_RESPONSE_MODEL
    ).station_response(TIME, frequencies, off_axis, direction)[0, :2]

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, unflagged)


def test_should_scale_station_response_for_telescope(telescope, direction):
    frequencies = np.array(
        [
            1.0005e5,
            1.0010e5,
            1.0015e5,
            1.0020e5,
        ]
    )

    scale = (
        np.ones(
            (frequencies.size,),
            dtype=frequencies.dtype,
        )
        / 2.0
    )

    assert telescope.type is eb.OSKAR

    beam_create_telescope = telescope.station_response(
        TIME, frequencies, direction, direction
    )

    beam_create_telescope_scaled = telescope.station_response(
        TIME,
        frequencies,
        direction,
        direction,
        scale,
    )

    np.testing.assert_allclose(
        beam_create_telescope / 2.0,
        beam_create_telescope_scaled,
    )


def test_should_validate_station_response_for_single_station(
    eb_ms, telescope, direction
):
    frequency = 1.0005e5

    loaded_telescope = eb.load_telescope(
        eb_ms, element_response_model=ELEMENT_RESPONSE_MODEL
    )

    np.testing.assert_allclose(
        loaded_telescope.station_response(
            TIME, 0, frequency, direction, direction
        ),
        telescope.station_response(
            TIME,
            frequency,
            direction,
            direction,
            station_idx=0,
        ).squeeze(axis=(0, 1, 2)),
    )


def test_should_pickle_initialised_telescope(telescope):
    assert telescope.type is eb.OSKAR

    unpickled = pickle.loads(pickle.dumps(telescope))

    assert telescope._telescope is not None
    assert unpickled._telescope is None
    assert unpickled.type is eb.OSKAR
    assert unpickled._nstations == telescope._nstations


def test_should_initialise_shared_telescope_once_across_threads(
    telescope, direction
):
    init_calls = []
    original_init = telescope._initialize_everybeam_telescope

    def counting_init():
        init_calls.append(1)
        original_init()

    telescope._initialize_everybeam_telescope = counting_init

    frequencies = np.array([1.0e8, 1.1e8])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: telescope.station_response(
                    TIME, frequencies, direction, direction
                ),
                range(16),
            )
        )

    assert len(init_calls) == 1
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
