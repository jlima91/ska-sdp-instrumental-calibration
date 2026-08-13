# pylint: skip-file
# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
from astropy.time import Time
from everybeam import OSKAR
from mock import MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.data_managers.beams import (
    BeamsFactory,
    BeamsLow,
    PointingBelowHorizon,
    convert_time_to_solution_time,
)


@patch("ska_sdp_instrumental_calibration.data_managers.beams.AltAz")
@patch("ska_sdp_instrumental_calibration.data_managers.beams.eb")
@patch("ska_sdp_instrumental_calibration.data_managers.beams.radec_to_xyz")
@patch(
    "ska_sdp_instrumental_calibration.data_managers.beams.convert_time_to_solution_time"
)
def test_initialise_beams(
    convert_time_mock, radec_to_xyz_mock, everybeam_mock, altaz_mock
):
    mock_array_location = Mock(name="array_location")
    mock_telescope = Mock(name="telescope")

    mock_direction = Mock(name="direction")
    altaz = Mock(name="altaz")
    altaz.alt.degree = 40
    mock_direction.transform_to.return_value = altaz

    mock_time = Mock(name="time")
    mock_time.mjd = 60000.0
    convert_time_mock.return_value = mock_time

    mock_telescope = Mock()
    everybeam_mock.load_telescope.return_value = mock_telescope

    frequency = np.array([100e6, 110e6])

    beams_low = BeamsLow(
        array_location=mock_array_location,
        direction=mock_direction,
        frequency=frequency,
        soln_time=123,
        telescope=mock_telescope,
    )

    assert beams_low.telescope == mock_telescope
    assert beams_low.frequency.size == 2
    convert_time_mock.assert_called_once_with(123)
    radec_to_xyz_mock.assert_called_once_with(mock_direction, mock_time)


@patch("ska_sdp_instrumental_calibration.data_managers.beams.AltAz")
@patch("ska_sdp_instrumental_calibration.data_managers.beams.SkyCoord")
@patch("ska_sdp_instrumental_calibration.data_managers.beams.radec_to_xyz")
@patch(
    "ska_sdp_instrumental_calibration.data_managers.beams.convert_time_to_solution_time"
)
def test_oskar_scale_computation(
    convert_time_mock,
    radec_to_xyz_mock,
    sky_coord_mock,
    altaz_mock,
):

    mock_time = Mock(name="time")
    mock_time.mjd = 60000.0
    convert_time_mock.return_value = mock_time

    mock_sky_coord = MagicMock(name="sky coord")
    sky_coord_mock.return_value = mock_sky_coord

    mock_array_location = Mock()
    mock_direction = Mock()

    altaz = Mock()
    altaz.alt.degree = 45
    mock_direction.transform_to.return_value = altaz

    radec_to_xyz_mock.return_value = np.array([1.0, 0.0, 0.0])

    mock_telescope = MagicMock(name="telescope")
    mock_telescope.single_station_response.return_value = np.ones((2, 2))
    mock_telescope.type = OSKAR

    freqs = np.array([100e6, 200e6, 300e6])

    bl = BeamsLow(
        array_location=mock_array_location,
        direction=mock_direction,
        frequency=freqs,
        soln_time=1.0,
        telescope=mock_telescope,
    )

    expected_scales = np.sqrt(2) / 2

    np.testing.assert_allclose(bl.scale, expected_scales, atol=1e-7)
    radec_to_xyz_mock.assert_has_calls(
        [call(mock_direction, mock_time), call(mock_sky_coord, mock_time)]
    )
    sky_coord_mock.assert_called_once_with(
        alt=90,
        az=0,
        unit="deg",
        frame="altaz",
        obstime=mock_time,
        location=mock_array_location,
    )
    assert mock_telescope.single_station_response.call_count == len(freqs)


@patch("ska_sdp_instrumental_calibration.data_managers.beams.AltAz")
@patch("ska_sdp_instrumental_calibration.data_managers.beams.radec_to_xyz")
@patch(
    "ska_sdp_instrumental_calibration.data_managers.beams.convert_time_to_solution_time"
)
def test_pointing_below_horizon(
    convert_time_mock, radec_to_xyz_mock, altaz_mock
):
    mock_time = Mock(name="time")
    mock_time.mjd = 60000.0
    convert_time_mock.return_value = mock_time

    mock_array_location = Mock(name="array location")
    mock_direction = Mock(name="direction")

    altaz = Mock(name="altaz")
    altaz.alt.degree = -10
    mock_direction.transform_to.return_value = altaz

    radec_to_xyz_mock.return_value = np.array([1.0, 0.0, 0.0])

    mock_telescope = MagicMock(name="telescope")

    freqs = np.array([100e6, 200e6, 300e6])

    with pytest.raises(PointingBelowHorizon, match="Pointing below horizon"):
        BeamsLow(
            array_location=mock_array_location,
            direction=mock_direction,
            frequency=freqs,
            soln_time=1.0,
            telescope=mock_telescope,
        )


@patch("ska_sdp_instrumental_calibration.data_managers.beams.AltAz")
@patch("ska_sdp_instrumental_calibration.data_managers.beams.radec_to_xyz")
@patch(
    "ska_sdp_instrumental_calibration.data_managers.beams.convert_time_to_solution_time"
)
def test_array_response(
    convert_time_mock,
    radec_to_xyz_mock,
    altaz_mock,
):
    mock_time = Mock(name="time", mjd=60000.0)
    convert_time_mock.return_value = mock_time

    altaz = Mock(name="altaz")
    altaz.alt.degree = 45

    mock_direction = Mock(name="direction")
    mock_direction.transform_to.return_value = altaz

    mock_telescope = MagicMock()

    bl = BeamsLow(
        array_location=Mock(name="location"),
        direction=mock_direction,
        frequency=np.array([100e6, 200e6]),
        soln_time=1.0,
        telescope=mock_telescope,
    )

    bl.scale = np.array([0.5, 0.25])

    beams = bl.array_response(Mock())

    mock_telescope.station_response.assert_called_once_with(
        solution_time=bl.solution_time_mjd_seconds,
        frequencies=bl.frequency,
        station0=radec_to_xyz_mock.return_value,
        tile0=bl.delay_dir_itrf,
        scale=bl.scale,
    )

    assert beams == mock_telescope.station_response.return_value


@patch("ska_sdp_instrumental_calibration.data_managers.beams.BeamsLow")
def test_beams_factory_get_beams_low(beams_low_mock):

    mock_telescope = Mock(name="telescope")
    factory = BeamsFactory(
        array_location=Mock(name="array_location"),
        direction=Mock(name="direction"),
        telescope=mock_telescope,
    )

    frequency = Mock(name="frequency")
    soln_time = 12345.0

    result = factory.get_beams_low(frequency, soln_time)

    beams_low_mock.assert_called_once_with(
        array_location=factory.array_location,
        direction=factory.direction,
        telescope=mock_telescope,
        frequency=frequency,
        soln_time=soln_time,
    )


def test_convert_time_to_solution_time():
    time_seconds = 60000.0 * 86400.0

    result = convert_time_to_solution_time(time_seconds)

    assert isinstance(result, Time)
    np.testing.assert_equal(result.mjd, 60000.0)
