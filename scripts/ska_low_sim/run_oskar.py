#!/usr/bin/env python3

"""
Run AA2-Low simulations for PI27 Low Goal 4.
Relates to feature SP-5416.

Authored by:
- Fred Dulwich
- Team Dhruva

This script requires the GLEAM sky model to be available in the current
working directory as a FITS catalogue called "GLEAM_EGC.fits".

Also required in the current directory is a YAML file called "fields.yaml",
which must contain the field and target information, for example:

---
fields:
  EoR2:         # MWA EoR2 field
    Cal1:       # Example bandpass calibrator (3C283)
      ra_deg: 197.914612
      dec_deg: -22.277973
      scan_id_start: 300
      transit_time: "2000-01-03 22:33:30.000"
    Centre:     # Target coordinates of centre of field
      ra_deg: 170.0
      dec_deg: -10.0
      scan_id_start: 400
      transit_time: "2000-01-04 20:38:00.000"

The field name is given using the --field argument, and
the target name within the field is given using the --target argument.
Both names must exist in the YAML file in the correct hierarchy.
For each target, the keys "ra_deg", "dec_deg", "scan_id_start"
and "transit_time" must be specified.
Observations of targets will be centred around their transit time.

Telescope models must also be present, passed to the --tel-model option.

Example script command line:
$ python3 run_oskar.py --output-dir product/model --field EoR2 --target Cal1 --num-scans 1 --scan-index 0 --tel-model path/to/telmodel --obs-length-mins 5 --make-image --make-psf
"""

import random

# Setting seed to a fixed value in order to achieve repeatability of results.
random.seed(100)

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
from astropy.io import fits
import numpy
import oskar
import yaml


def filter_sky_model(
    sky: oskar.Sky,
    fraction: float,
    inner_radius_deg: float,
    outer_radius_deg: float,
    ra0_deg: float,
    dec0_deg: float,
):
    """
    Filter a sky model to keep only the brightest sources in the given region.

    :param sky: The input sky model.
    :param fraction: Normalised fraction of brightest sources to return.
    :param inner_radius_deg: Inner radius, in degrees.
    :param outer_radius_deg: Outer radius, in degrees.
    :param ra0_deg: Central RA, in degrees.
    :param dec0_deg: Central Dec, in degrees.

    :return: The filtered sky model.
    """
    tmp = sky.create_copy()
    tmp.filter_by_radius(inner_radius_deg, outer_radius_deg, ra0_deg, dec0_deg)
    num_sources = tmp.num_sources
    sky_array = tmp.to_array()

    # Sort by flux, descending.
    num_to_keep = int(fraction * num_sources)
    sky_sorted = sky_array[sky_array[:, 2].argsort()[::-1]]

    # Keep only the brightest sources.
    sky_sorted = sky_sorted[0:num_to_keep, :]
    return oskar.Sky.from_array(sky_sorted)


def gleam_sky_model(ra0_deg: float, dec0_deg: float, max_radius_deg: float):
    """
    Generate a sky model from the GLEAM catalogue.

    :param ra0_deg: Central RA, in degrees.
    :param dec0_deg: Central Dec, in degrees.
    :param max_radius_deg: Maximum radius to keep.

    :return: The GLEAM-based sky model.
    """
    data = fits.getdata("GLEAM_EGC.fits", 1)
    stokes_I = data["int_flux_143"]
    stokes_Q = numpy.zeros_like(stokes_I)
    stokes_U = numpy.zeros_like(stokes_I)
    stokes_V = numpy.zeros_like(stokes_I)
    ref_freq_hz = numpy.ones_like(stokes_I) * 143e6
    alpha = data["alpha"]
    sky_array = numpy.column_stack(
        (
            data["RAJ2000"],
            data["DEJ2000"],
            stokes_I,
            stokes_Q,
            stokes_U,
            stokes_V,
            ref_freq_hz,
            alpha,
        )
    )
    # Remove sources with NaN in flux column.
    sky_array = sky_array[~numpy.isnan(sky_array[:, 2])]
    # Remove sources with negative fluxes (yes, there are about 3000 of those!).
    sky_array = sky_array[sky_array[:, 2] > 0]
    # Replace NaNs in spectral index column with zeros.
    sky_array = numpy.nan_to_num(sky_array)
    gleam = oskar.Sky.from_array(sky_array)
    gleam.filter_by_radius(0, max_radius_deg, ra0_deg, dec0_deg)
    # Keep everything inside 2.5 degrees.
    band0 = filter_sky_model(gleam, 1.0, 0.0, 2.5, ra0_deg, dec0_deg)
    # Keep only the brightest 50% of sources between 2.5 and 5 degrees.
    band1 = filter_sky_model(gleam, 0.5, 2.5, 5.0, ra0_deg, dec0_deg)
    # Keep only the brightest 10% of sources outside 5 degrees.
    band2 = filter_sky_model(gleam, 0.1, 5.0, max_radius_deg, ra0_deg, dec0_deg)
    sky_composite = band0.create_copy()
    sky_composite.append(band1)
    sky_composite.append(band2)
    return sky_composite


def main():
    """
    Main function to run simulations.
    """
    # Get command line arguments.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If specified, stop short of actually running the simulation",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument(
        "--field", required=True, help="Field name in accompanying YAML file"
    )
    parser.add_argument(
        "--target", required=True, help="Target name in field (from YAML file)"
    )
    parser.add_argument(
        "--tec-screen", default="", help="Path to TEC screens FITS image"
    )
    parser.add_argument(
        "--num-scans", type=int, default=1, help="No. scans in observation"
    )
    parser.add_argument("--scan-index", type=int, default=0, help="Scan index to run")
    parser.add_argument(
        "--field-radius-deg", type=float, default=10.0, help="Field radius"
    )
    parser.add_argument(
        "--tel-model", required=True, help="Path to the telescope model directory"
    )
    parser.add_argument(
        "--obs-length-mins", type=float, default=240.0, help="Length (minutes)"
    )
    parser.add_argument(
        "--dump-time-sec", type=float, default=0.84934656, help="Dump time (s)"
    )
    parser.add_argument(
        "--start-freq-hz", type=float, default=100e6, help="Start freq (Hz)"
    )
    parser.add_argument(
        "--end-freq-hz", type=float, default=175e6, help="End freq (Hz)"
    )
    parser.add_argument(
        "--max-sources-per-chunk", type=int, default=2000, help="Chunk size"
    )
    parser.add_argument(
        "--channel-width-hz",
        type=float,
        default=5.425347222222222e3,
        help="Channel width (Hz)",
    )
    parser.add_argument(
        "--add-gleam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If specified, add GLEAM components to the sky model",
    )
    parser.add_argument(
        "--add-centaurus-a",
        action="store_true",
        default=False,
        help="If specified, add Centaurus A to the sky model",
    )
    parser.add_argument(
        "--add-fornax-a",
        action="store_true",
        default=False,
        help="If specified, add Fornax A to the sky model",
    )
    parser.add_argument(
        "--add-taurus-a",
        action="store_true",
        default=False,
        help="If specified, add Taurus A to the sky model",
    )
    parser.add_argument(
        "--make-image",
        action="store_true",
        default=False,
        help="If specified, make a dirty image",
    )
    parser.add_argument(
        "--make-psf",
        action="store_true",
        default=False,
        help="If specified, make an image of the PSF",
    )
    parser.add_argument(
        "--image-size", type=int, default=2980, help="Image size (pixels)"
    )
    parser.add_argument(
        "--cellsize-arcsec", type=float, default=21.66, help="Image pixel size (arcsec)"
    )
    parser.add_argument(
        "--double-precision",
        action="store_true",
        default=False,
        help="Whether to use double precision for oskar computations. Default False, i.e. use single precision. Note that MSv2 data is always in single precision.",
    )
    parser.add_argument(
        "--use-gpus",
        action="store_true",
        default=False,
        help="Whether to use GPU. Default False.",
    )
    args = parser.parse_args()

    # Print arguments.
    print("Script argument list:")
    print("\n".join(f"    {k}={v}" for k, v in vars(args).items()))

    # Load the known fields and targets.
    with open("fields.yaml") as stream:
        fields = yaml.safe_load(stream)["fields"]

    # Check field and target names are known.
    if args.field not in fields:
        raise RuntimeError(f"Field name '{args.field}' not known")
    if args.target not in fields[args.field]:
        raise RuntimeError(f"Target name '{args.target}' not known")

    # Calculate start time for the scan.
    transit_time_str = fields[args.field][args.target]["transit_time"]
    transit_time = datetime.fromisoformat(transit_time_str)
    half_obs_length_mins = 0.5 * args.obs_length_mins
    scan_length_mins = args.obs_length_mins / args.num_scans
    scan_offset_mins = args.scan_index * scan_length_mins
    start_time = transit_time + (
        timedelta(minutes=scan_offset_mins) - timedelta(minutes=half_obs_length_mins)
    )

    # Find telescope model name to use, and check that it exists.
    tel_model = args.tel_model
    if not os.path.isdir(tel_model):
        raise RuntimeError(f"Telescope model {tel_model} does not exist!")

    # Generate the output directory path.
    date_str = f"{start_time.year}{start_time.month:02}{start_time.day:02}"
    output_dir = args.output_dir

    # Generate the output MS name.
    scan_id_start = fields[args.field][args.target]["scan_id_start"]
    ms_path = os.path.join(
        output_dir, f"visibility.scan-{args.scan_index + scan_id_start}.ms"
    )

    # Set up remaining simulation parameters.
    ra_deg = fields[args.field][args.target]["ra_deg"]
    dec_deg = fields[args.field][args.target]["dec_deg"]
    num_times = int(numpy.floor(60 * scan_length_mins / args.dump_time_sec))
    num_channels = int(
        numpy.floor((args.end_freq_hz - args.start_freq_hz) / args.channel_width_hz)
    )
    print(f"Running scan {args.scan_index + 1} / {args.num_scans}")
    print(f"Scan start time: {str(start_time)}")
    print(f"Scan length (minutes): {scan_length_mins}")
    print(f"Using telescope model (exists): {tel_model}")
    print(f"Number of time samples in scan: {num_times}")
    print(f"Number of channels: {num_channels}")
    print(f"Output Measurement Set: {ms_path}")
    sim_params = {
        "simulator/double_precision": args.double_precision,
        "simulator/use_gpus": args.use_gpus,
        "simulator/max_sources_per_chunk": args.max_sources_per_chunk,
        "simulator/keep_log_file": True,
        "simulator/write_status_to_log_file": True,
        "observation/phase_centre_ra_deg": ra_deg,
        "observation/phase_centre_dec_deg": dec_deg,
        "observation/start_frequency_hz": args.start_freq_hz,
        "observation/num_channels": num_channels,
        "observation/frequency_inc_hz": args.channel_width_hz,
        "observation/start_time_utc": str(start_time),
        "observation/length": 60 * scan_length_mins,
        "observation/num_time_steps": num_times,
        "telescope/input_directory": tel_model,
        "telescope/normalise_beams_at_phase_centre": False,
        "telescope/aperture_array/array_pattern/normalise": True,
        "interferometer/channel_bandwidth_hz": args.channel_width_hz,
        "interferometer/time_average_sec": args.dump_time_sec,
        "interferometer/max_time_samples_per_block": 8,
        "interferometer/max_channels_per_block": num_channels,
        "interferometer/ms_filename": ms_path,
        "interferometer/ms_dish_diameter": 38,
    }
    if args.tec_screen:
        tec_screen = args.tec_screen
        if not os.path.isfile(tec_screen):
            raise RuntimeError(f"TEC screen {tec_screen} does not exist!")
        print(f"Using TEC screen (exists): {tec_screen}")
        sim_params["telescope/ionosphere_screen_type"] = "External"
        sim_params["telescope/external_tec_screen/input_fits_file"] = tec_screen

    # Set up the sky model.
    sky_composite = oskar.Sky()
    # Create a GLEAM-based sky model if specified.
    if args.add_gleam:
        sky_gleam = gleam_sky_model(ra_deg, dec_deg, args.field_radius_deg)
        sky_composite.append(sky_gleam)

    # Add bright sources if specified.
    # Fornax A: data from the Molonglo Southern 4 Jy sample (VizieR).
    # Others from GLEAM reference paper, Hurley-Walker et al. (2017), Table 2.
    if args.add_centaurus_a:
        print("Adding Centaurus A to sky model")
        sky_centaurus = oskar.Sky.from_array(
            [201.36667, -43.01917, 1370, 0, 0, 0, 200e6, -0.50, 0, 0, 0, 0]
        )
        sky_composite.append(sky_centaurus)
    if args.add_fornax_a:
        print("Adding Fornax A to sky model")
        sky_fornax = oskar.Sky.from_array(
            [50.67375, -37.20833, 528, 0, 0, 0, 178e6, -0.51, 0, 0, 0, 0]
        )
        sky_composite.append(sky_fornax)
    if args.add_taurus_a:
        print("Adding Taurus A to sky model")
        sky_taurus = oskar.Sky.from_array(
            [83.63333, 22.01444, 1340, 0, 0, 0, 200e6, -0.22, 0, 0, 0, 0]
        )
        sky_composite.append(sky_taurus)
    print(f"Sky model contains {sky_composite.num_sources} sources.")

    # Stop now if doing a dry run.
    if args.dry_run:
        print("Dry run specified: Exiting.")
        return

    # Ensure the output directory exists.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create the sky model and region file if necessary.
    sky_path_csv = os.path.join(output_dir, f"sky_model.csv")
    sky_path_reg = os.path.join(output_dir, f"sky_model.reg")
    if not os.path.isfile(sky_path_csv):
        sky_composite.save(sky_path_csv)
    if not os.path.isfile(sky_path_reg):
        sky_composite.to_ds9_regions(sky_path_reg)

    # Run simulation.
    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(sim_params)
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky_composite)
    sim.run()

    # Make dirty image if required.
    if args.make_image:
        img_params = {
            "image/size": args.image_size,
            "image/specify_cellsize": True,
            "image/cellsize_arcsec": args.cellsize_arcsec,
            "image/algorithm": "W-projection",
            "image/fft/use_gpu": True,
            "image/fft/grid_on_gpu": True,
            "image/input_vis_data": ms_path,
            "image/root_path": os.path.splitext(ms_path)[0],
        }
        settings = oskar.SettingsTree("oskar_imager")
        settings.from_dict(img_params)
        img = oskar.Imager(settings=settings)
        img.run()

    # Make PSF if required.
    if args.make_psf:
        img_params = {
            "image/image_type": "PSF",
            "image/size": args.image_size,
            "image/specify_cellsize": True,
            "image/cellsize_arcsec": args.cellsize_arcsec,
            "image/algorithm": "W-projection",
            "image/fft/use_gpu": True,
            "image/fft/grid_on_gpu": True,
            "image/input_vis_data": ms_path,
            "image/root_path": os.path.splitext(ms_path)[0],
        }
        settings = oskar.SettingsTree("oskar_imager")
        settings.from_dict(img_params)
        img = oskar.Imager(settings=settings)
        img.run()


if __name__ == "__main__":
    main()
