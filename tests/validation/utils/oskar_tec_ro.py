"""
Oskar tec screen fits file simulation script written for
- [PI29] Investigating ionosphere representativeness

author: Vijay Mahatma <vm462@cam.ac.uk>

Generate a test ionsopheric screen with multiple layers.
ARatmospy must be in the PYTHONPATH https://github.com/shrieks/ARatmospy

A python3 compatible version of ARatmospy is available in this fork
: https://github.com/abhinavn17/ARatmospy
"""

import logging

import numpy
import numpy as np
from ArScreens import ArScreens  # pylint: disable=import-error
from astropy.io import fits
from astropy.wcs import WCS

from .constants import RANDOM_SEED

np.random.seed(RANDOM_SEED)

logger = logging.getLogger("OSKAR TEC screen generation")
logging.basicConfig(level=logging.INFO)


def run_tec_screens(tec_file_path):

    screen_width_metres = 200e3
    r0 = 20e3  # Scale size (5 km).
    bmax = 20e3  # 20 km sub-aperture size.
    sampling = 100.0  # 100 m/pixel.
    m = int(bmax / sampling)  # Pixels per sub-aperture (200).
    n = int(
        screen_width_metres / bmax
    )  # Sub-apertures across the screen (10).
    num_pix = n * m
    pscale = screen_width_metres / (n * m)  # Pixel scale (100 m/pixel).
    logger.info(f"\nNumber of pixels {num_pix:d}, pixel size {pscale:.3f} m")
    logger.info(f"Field of view {num_pix * pscale:.1f} (m)")
    speed = 150e3 / 3600.0  # 150 km/h in m/s.
    # Parameters for each layer.
    # (scale size [m], speed [m/s], direction [deg], layer height [m]).
    layer_params = numpy.array(
        [(r0, speed, 60.0, 300e3), (r0, speed / 2.0, -30.0, 310e3)]
    )

    rate = 5.0 / 60.0  # The inverse frame rate (1 per minute).
    alpha_mag = 0.999  # Evolve screen slowly.
    num_times = 6  # Four hours.
    my_screens = ArScreens(n, m, pscale, rate, layer_params, alpha_mag)
    logger.info("Running screens...")
    my_screens.run(num_times, verbose=False)
    logger.info("Done")
    # Convert to TEC
    # phase = image[pixel] * -8.44797245e9 / frequency
    frequency = 1e8
    phase2tec = -frequency / 8.44797245e9

    data = numpy.zeros([1, num_times, num_pix, num_pix])
    for layer in range(len(my_screens.screens)):
        for i, screen in enumerate(my_screens.screens[layer]):
            data[:, i, ...] += phase2tec * screen[numpy.newaxis, ...]

    # Check TEC rms
    tec_rms = []
    tec_ptp = []

    for i in range(num_times):
        tec = data[0, i, :, :]
        tec -= tec.mean()
        tec_rms.append(np.std(tec))
        tec_ptp.append(tec.max() - tec.min())

    tec_rms = np.array(tec_rms)

    logger.info(
        f"TEC RMS: min/mean/max = "
        f"{np.min(tec_rms):.4f} / "
        f"{np.mean(tec_rms):.4f} / "
        f"{np.max(tec_rms):.4f} TECU"
    )

    logger.info(f"TEC peak-to-peak (mean): " f"{np.mean(tec_ptp):.4f} TECU")
    # Re-scale TEC amplitude to physical RMS

    target_rms = 0.01  # TECU (quiet to moderate conditions)
    logger.info(f"\nRescaling TEC RMS to {target_rms:.4f} TECU")
    data *= target_rms / tec_rms.mean()

    # Now verify values after rescaling

    tec_rms_after = []

    for i in range(num_times):
        tec = data[0, i]
        tec -= tec.mean()
        tec_rms_after.append(np.std(tec))

    tec_rms_after = np.array(tec_rms_after)

    logger.info("\n---TEC statistics after rescaling ---")

    logger.info(
        f"RMS TEC: min/mean/max = "
        f"{tec_rms_after.min():.4f} / "
        f"{tec_rms_after.mean():.4f} / "
        f"{tec_rms_after.max():.4f} TECU"
    )

    w = WCS(naxis=4)
    w.naxis = 4
    w.wcs.cdelt = [pscale, pscale, 1.0 / rate, 1.0]
    w.wcs.crpix = [num_pix // 2 + 1, num_pix // 2 + 1, num_times // 2 + 1, 1.0]
    w.wcs.ctype = ["XX", "YY", "TIME", "FREQ"]
    w.wcs.crval = [0.0, 0.0, 0.0, frequency]

    fits.writeto(
        filename=tec_file_path,
        data=data,
        header=w.to_header(),
        overwrite=True,
    )
