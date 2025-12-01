from functools import wraps
from traceback import print_exc

import numpy as np

from ska_sdp_instrumental_calibration.logger import setup_logger

logger = setup_logger(__name__)


def safe(func):
    """
    Wrapper to catch all exceptions and print traceback to stderr,
    instead of crashing the application.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as ex:
            logger.error(
                "Caught exception in function %s: %s", func.__name__, str(ex)
            )
            print_exc()

    return wrapper


def _ecef_to_lla(x, y, z):
    """Translate Earth-Centred Earth-Fixed (in meters) to
    geodetic (WGS84) coordinates.

    Parameters
    ----------
    x : array of position(s) on x-axis in meters
    y : array of position(s) on y-axis in meters
    z : array of position(s) on z-axis in meters

    Returns
    -------
    Single set/array of latitude(s), longitude(s) in radians or decimal degrees
    and elevation(s) in meters.

    Notes
    -----
    Based on Vermeille, H. Journal of Geodesy (2002) 76: 451
    (https://doi.org/10.1007/s00190-002-0273-6) and
    Vermeille, H. Journal of Geodesy (2004) 78: 94
    (https://doi.org/10.1007/s00190-004-0375-4).
    """
    equatorial_radius = 6378137.0  # Semi-major axis of the Earth (in meters).
    polar_radius = 6356752.314245179
    flattening_factor = (
        equatorial_radius - polar_radius
    ) / equatorial_radius  # Flattening of the reference ellipsoid.
    p = (x * x + y * y) / (equatorial_radius * equatorial_radius)
    esq = flattening_factor * (2.0 - flattening_factor)
    q = (1.0 - esq) / (equatorial_radius * equatorial_radius) * z * z
    r = (p + q - esq * esq) / 6.0
    s = esq * esq * p * q / (4 * r * r * r)
    t = np.power(1.0 + s + np.sqrt(s * (2.0 + s)), 1.0 / 3.0)
    u = r * (1.0 + t + 1.0 / t)
    v = np.sqrt(u * u + esq * esq * q)
    w = esq * (u + v - q) / (2.0 * v)
    k = np.sqrt(u + v + w * w) - w
    D = k * np.sqrt(x * x + y * y) / (k + esq)
    altitude = (k + esq - 1.0) / k * np.sqrt(D * D + z * z)
    latitude = np.rad2deg(2.0 * np.arctan2(z, D + np.sqrt(D * D + z * z)))

    sign = 1 if np.any(y) >= 0.0 else -1
    longitude = np.rad2deg(
        sign * 0.5 * np.pi
        - sign * 2.0 * np.arctan2(x, np.sqrt(x * x + y * y) + (sign * y))
    )
    return latitude, longitude, altitude
