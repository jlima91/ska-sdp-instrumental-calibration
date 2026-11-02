#!/usr/bin/env python3
"""
TEC screen generation script for OSKAR simulations.

Based on the OSKAR TEC screen script written for [PI29] Investigating
ionosphere representativeness.

Authored by:
- Vijay Mahatma
- Team Dhruva

Generates an ionospheric TEC screen with multiple layers (frozen flow with
autoregressive turbulence), with ARatmospy. The screen is written as a FITS
file which OSKAR reads as an external TEC screen
(simulate_visibility.tec_screen).

The parameters are read from a YAML config file (sim.yaml):
- The "generate_tec" section: layers, screen size and resolution, target
  TEC RMS, output file.
- The common section: the screen covers the observation (observing_time_mins),
  with one screen per sampling_time_sec (unless generate_tec.time_interval_sec
  is set), and its reference frequency is the centre of the band. The screen
  is checked to cover the field of the first target (see
  check_screen_coverage).

ARatmospy is not a Python package. Use the Python 3 fork
https://github.com/abhinavn17/ARatmospy, and set generate_tec.aratmospy_dir
to its folder, or add it to PYTHONPATH.

Usage:
python generate_tec.py sim.yaml
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from lib.sim_config import GenerateTecConfig, SimConfig

# Phase (radians) of 1 TECU at 1 Hz: phase = -8.44797245e9 * TEC / frequency
PHASE_PER_TECU_HZ = 8.44797245e9

# Height of the TEC screen used by OSKAR (telescope/external_tec_screen/
# screen_height_km, not set by simulate_visibility.py, default 300 km)
OSKAR_SCREEN_HEIGHT_M = 300e3

EARTH_RADIUS_M = 6371e3


@dataclass
class ScreenAxes:
    """Pixel, time and frequency axes of the TEC screen."""

    n_subapertures: int
    pixels_per_subaperture: int
    pixel_size_m: float
    time_interval_sec: float
    n_times: int
    frequency_hz: float

    @property
    def n_pixels(self) -> int:
        return self.n_subapertures * self.pixels_per_subaperture

    @classmethod
    def from_config(cls, cfg: SimConfig, tec_cfg: GenerateTecConfig) -> "ScreenAxes":
        n_subapertures = int(tec_cfg.screen_width_m / tec_cfg.subaperture_size_m)
        pixels_per_subaperture = int(tec_cfg.subaperture_size_m / tec_cfg.pixel_size_m)
        time_interval_sec = tec_cfg.time_interval_sec or cfg.sampling_time_sec
        # One more screen than intervals, so that the last time is covered
        n_times = math.ceil(cfg.observing_time_mins * 60 / time_interval_sec) + 1
        return cls(
            n_subapertures=n_subapertures,
            pixels_per_subaperture=pixels_per_subaperture,
            # The screen width is kept: if it is not a whole number of
            # sub-apertures and pixels, the pixels are made larger
            pixel_size_m=tec_cfg.screen_width_m
            / (n_subapertures * pixels_per_subaperture),
            time_interval_sec=time_interval_sec,
            n_times=n_times,
            frequency_hz=0.5
            * (cfg.simulation_start_frequency_hz + cfg.simulation_end_frequency_hz),
        )


def import_ar_screens(aratmospy_dir: Optional[str]):
    """Import ArScreens, from aratmospy_dir if set."""
    if aratmospy_dir:
        if not os.path.isdir(aratmospy_dir):
            raise FileNotFoundError(
                f"generate_tec.aratmospy_dir not found: {aratmospy_dir}"
            )
        # ArScreens imports the other ARatmospy modules as top level modules
        sys.path.insert(0, os.path.abspath(aratmospy_dir))
    try:
        from ArScreens import ArScreens  # pylint: disable=import-error
    except ImportError as err:
        raise ImportError(
            f"Cannot import ArScreens ({err}). Clone "
            "https://github.com/abhinavn17/ARatmospy, and set "
            "generate_tec.aratmospy_dir to its folder (or add it to PYTHONPATH)."
        ) from None
    return ArScreens


def array_radius_m(tel_model: str) -> Optional[float]:
    """
    Largest distance (m) of a station from the telescope position, or None
    if the telescope model has no layout_wgs84.txt / position.txt.
    """
    layout_file = Path(tel_model) / "layout_wgs84.txt"
    position_file = Path(tel_model) / "position.txt"
    if not (layout_file.is_file() and position_file.is_file()):
        return None
    lon0, lat0 = np.loadtxt(position_file, ndmin=1)[:2]
    lon, lat = np.loadtxt(layout_file, ndmin=2)[:, :2].T
    # Small angle approximation, good enough for a few 10 km
    dx = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
    dy = np.radians(lat - lat0) * EARTH_RADIUS_M
    return float(np.max(np.hypot(dx, dy)))


def telescope_latitude_deg(tel_model: str) -> Optional[float]:
    position_file = Path(tel_model) / "position.txt"
    if not position_file.is_file():
        return None
    return float(np.loadtxt(position_file, ndmin=1)[1])


def check_screen_coverage(cfg: SimConfig, screen_width_m: float):
    """
    Warn if the screen is too small for the field of the first target.

    OSKAR centres the screen on the telescope, at OSKAR_SCREEN_HEIGHT_M.
    The pierce point of a direction at zenith angle za is at
    height * tan(za) from the telescope. The field (of radius
    simulate_visibility.field_radius_deg) is taken at the transit of the
    target, where its zenith angle is |dec - latitude|.
    """
    field_name, target_name, target = cfg.first_target()
    if cfg.simulate_visibility is None:
        print("Cannot check the screen coverage: no simulate_visibility.tel_model")
        return
    tel_model = cfg.simulate_visibility.tel_model
    latitude = telescope_latitude_deg(tel_model)
    radius = array_radius_m(tel_model)
    if latitude is None or radius is None:
        print(f"Cannot check the screen coverage: no position in {tel_model}")
        return

    field_radius_deg = cfg.simulate_visibility.field_radius_deg
    zenith_angle_deg = abs(target.dec_deg - latitude)
    max_zenith_angle_deg = zenith_angle_deg + field_radius_deg
    if max_zenith_angle_deg >= 90.0:
        print(
            f"WARNING: the field of {field_name}/{target_name} reaches the "
            "horizon, the screen cannot cover it"
        )
        return
    required_width_m = 2 * (
        OSKAR_SCREEN_HEIGHT_M * math.tan(math.radians(max_zenith_angle_deg)) + radius
    )
    print(
        f"Target {field_name}/{target_name}: zenith angle at transit "
        f"{zenith_angle_deg:.1f} deg, field radius {field_radius_deg:.1f} deg. "
        f"Screen width needed: {required_width_m / 1e3:.1f} km"
    )
    if screen_width_m < required_width_m:
        print(
            f"WARNING: generate_tec.screen_width_m ({screen_width_m / 1e3:.1f} km) "
            "is smaller than needed, the field edge is outside the screen"
        )


def tec_rms_per_time(data: np.ndarray) -> np.ndarray:
    """Remove the mean of each screen (in place), and return their RMS."""
    rms = []
    for i in range(data.shape[1]):
        tec = data[0, i]
        tec -= tec.mean()
        rms.append(np.std(tec))
    return np.array(rms)


def print_rms(label: str, rms: np.ndarray):
    print(
        f"{label}: TEC RMS min/mean/max = {rms.min():.4f} / {rms.mean():.4f} / "
        f"{rms.max():.4f} TECU"
    )


def simulate_tec_screen(tec_cfg: GenerateTecConfig, axes: ScreenAxes) -> np.ndarray:
    """
    Simulate the screens of all the layers, and add them.

    Returns
    -------
    TEC (TECU), with shape (1, n_times, n_pixels, n_pixels), and an RMS of
    tec_cfg.target_rms_tecu
    """
    ArScreens = import_ar_screens(tec_cfg.aratmospy_dir)
    # (scale size [m], speed [m/s], direction [deg], layer height [m])
    layer_params = np.array(
        [
            (layer.r0_m, layer.speed_m_per_s, layer.direction_deg, layer.height_m)
            for layer in tec_cfg.layers
        ]
    )
    screens = ArScreens(
        axes.n_subapertures,
        axes.pixels_per_subaperture,
        axes.pixel_size_m,
        1.0 / axes.time_interval_sec,
        layer_params,
        tec_cfg.alpha_mag,
        ranseed=tec_cfg.random_seed,
    )
    print(f"Simulating {len(tec_cfg.layers)} layers...")
    screens.run(axes.n_times, verbose=False)

    # The screens are phases at the reference frequency
    phase_to_tec = -axes.frequency_hz / PHASE_PER_TECU_HZ
    data = np.zeros([1, axes.n_times, axes.n_pixels, axes.n_pixels])
    for layer_screens in screens.screens:
        for i, screen in enumerate(layer_screens):
            data[0, i] += phase_to_tec * screen

    rms = tec_rms_per_time(data)
    print_rms("Simulated", rms)
    data *= tec_cfg.target_rms_tecu / rms.mean()
    print_rms("Rescaled", tec_rms_per_time(data))
    return data


def write_tec_screen(path: str, data: np.ndarray, axes: ScreenAxes):
    """Write the screen in the FITS format read by OSKAR (ARatmospy format)."""
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # OSKAR reads the pixel size and the time interval from the header
    wcs = WCS(naxis=4)
    wcs.wcs.cdelt = [axes.pixel_size_m, axes.pixel_size_m, axes.time_interval_sec, 1.0]
    wcs.wcs.crpix = [
        axes.n_pixels // 2 + 1,
        axes.n_pixels // 2 + 1,
        axes.n_times // 2 + 1,
        1.0,
    ]
    wcs.wcs.ctype = ["XX", "YY", "TIME", "FREQ"]
    wcs.wcs.crval = [0.0, 0.0, 0.0, axes.frequency_hz]
    fits.writeto(path, data=data, header=wcs.to_header(), overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = SimConfig.load(args.config)
    tec_cfg = cfg.require_generate_tec()
    axes = ScreenAxes.from_config(cfg, tec_cfg)

    print(
        f"Screen: {axes.n_pixels} x {axes.n_pixels} pixels of "
        f"{axes.pixel_size_m:.1f} m, {axes.n_times} times every "
        f"{axes.time_interval_sec:.2f} s, reference frequency "
        f"{axes.frequency_hz / 1e6:.3f} MHz "
        f"({8 * axes.n_times * axes.n_pixels**2 / 1e9:.2f} GB)"
    )
    check_screen_coverage(cfg, axes.n_pixels * axes.pixel_size_m)

    start_time = time.perf_counter()
    data = simulate_tec_screen(tec_cfg, axes)
    write_tec_screen(tec_cfg.output_tec_screen, data, axes)
    print(f"Processing time: {time.perf_counter() - start_time:.0f} s")
    print(f"Wrote TEC screen to {tec_cfg.output_tec_screen}")


if __name__ == "__main__":
    main()
