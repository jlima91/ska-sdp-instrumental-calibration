import numpy as np
from mock import patch

from ska_sdp_instrumental_calibration.plot._util import ecef_to_lla, safe


@patch("ska_sdp_instrumental_calibration.plot._util.logger")
@patch("ska_sdp_instrumental_calibration.plot._util.print_exc")
def test_should_execute_wrapped_function_safely(
    print_exception_mock, logger_mock
):
    def add(a, b=100, c=10):
        return a + b + c

    wrapped_function = safe(add)

    result = wrapped_function(20, c=30)

    logger_mock.error.assert_not_called()
    print_exception_mock.assert_not_called()

    assert result == 150


@patch("ska_sdp_instrumental_calibration.plot._util.logger")
@patch("ska_sdp_instrumental_calibration.plot._util.print_exc")
def test_should_catch_exceptions_of_wrapped_function(
    print_exception_mock, logger_mock
):
    def unsafe_function(a, b=4):
        raise Exception(f"Got a: {a} and b: {b}")

    wrapped_function = safe(unsafe_function)

    wrapped_function(10)

    logger_mock.error.assert_called_once_with(
        "Caught exception in function %s: %s",
        "unsafe_function",
        "Got a: 10 and b: 4",
    )
    print_exception_mock.assert_called_once_with()


def test_should_convert_earth_centric_coordinates_to_geodetic():
    x = np.array([-5133977.79947732, -5133977.79947732])
    y = np.array([10168886.79974105, -10168886.79974105])
    z = np.array([-5723265.90488199, -5723265.90488199])

    lat, lng, alt = ecef_to_lla(x, y, z)

    np.testing.assert_allclose(lat, np.array([-26.753052, -26.753052]))
    np.testing.assert_allclose(lng, np.array([116.787894, 243.212106]))
    np.testing.assert_allclose(alt, np.array([6374502.632896, 6374502.632896]))
