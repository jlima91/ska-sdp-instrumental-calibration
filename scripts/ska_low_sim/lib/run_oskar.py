#!/usr/bin/env python3
"""
Run an SKA-Low OSKAR simulation of one scan of a target.

Based on the AA2-Low simulation script for PI27 Low Goal 4 (SP-5416).
This script is normally run by simulate_visibility.py, inside the OSKAR Singularity
image.

The simulation parameters are read from the YAML configuration file given
with --config (sim.yaml, see simulate_visibility.py):
- Frequency and time axes, and fields from the common section.
- Telescope model, sky models, station beam, smearing, OSKAR simulator options (precision,
  GPUs, noise), TEC screen, gain table, cable delays and OSKAR imager
  options from the "simulate_visibility" section.
The config file and the custom sky model are copied into "inputs" in the
output directory, and the TEC screen is linked there, as a record of the run.
The telescope model is copied into the output directory as "<name>.custom",
with the gain table and cable delays linked into it, and the simulation uses
the copy.
Relative paths in the config file are relative to the current directory
(where the script is started). The simulation then runs from --output-dir.
The options of this script are only for what is not in the config file.

The target is given with --field and --target (default: the first target of
the first field). Observations of targets are centred around their transit
time, and split into --num-scans scans. Only the scan --scan-index is
simulated.

Example script command line:
$ python3 run_oskar.py --config sim.yaml --output-dir product/model --field EoR2 --target Cal1
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy
import oskar
from astropy.io import fits

from sim_config import (
    MS_NAME_FORMAT,
    TEL_MODEL_CORRUPTION_FILES,
    OskarImagerConfig,
    SimConfig,
    SimulateVisibilityConfig,
    Target,
)

# Bright sources which can be added to the sky model
# (simulate_visibility.bright_sources), as OSKAR sky model rows
# (RA, Dec, I, Q, U, V, reference frequency, spectral index, RM, major,
# minor, PA).
# Fornax A: data from the Molonglo Southern 4 Jy sample (VizieR).
# Others from GLEAM reference paper, Hurley-Walker et al. (2017), Table 2.
BRIGHT_SOURCES = {
    "centaurus_a": [201.36667, -43.01917, 1370, 0, 0, 0, 200e6, -0.50, 0, 0, 0, 0],
    "fornax_a": [50.67375, -37.20833, 528, 0, 0, 0, 178e6, -0.51, 0, 0, 0, 0],
    "taurus_a": [83.63333, 22.01444, 1340, 0, 0, 0, 200e6, -0.22, 0, 0, 0, 0],
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_oskar",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If specified, stop short of actually running the simulation",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file (sim.yaml) with the simulation parameters. It "
        "must be readable inside the Singularity container",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument(
        "--field",
        default=None,
        help="Field name in the config file. Given with --target. Default: the "
        "field of the first target",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target name in the field. Given with --field. Default: the first "
        "target of the first field",
    )
    parser.add_argument(
        "--num-scans", type=int, default=1, help="No. scans in observation"
    )
    parser.add_argument("--scan-index", type=int, default=0, help="Scan index to run")
    parser.add_argument(
        "--max-sources-per-chunk", type=int, default=16384, help="Chunk size"
    )
    return parser.parse_args()


# ------------------------------ Sky model ------------------------------ #


def sky_from_array(sky_array, precision: str) -> oskar.Sky:
    return oskar.Sky(precision=precision).from_array(sky_array, precision=precision)


def filter_sky_model(
    sky: oskar.Sky,
    fraction: float,
    inner_radius_deg: float,
    outer_radius_deg: float,
    ra0_deg: float,
    dec0_deg: float,
    precision: str,
):
    """
    Filter a sky model to keep only the brightest sources in the given region.

    :param sky: The input sky model.
    :param fraction: Normalised fraction of brightest sources to return.
    :param inner_radius_deg: Inner radius, in degrees.
    :param outer_radius_deg: Outer radius, in degrees.
    :param ra0_deg: Central RA, in degrees.
    :param dec0_deg: Central Dec, in degrees.
    :param precision: OSKAR precision, "single" or "double".

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
    return sky_from_array(sky_sorted, precision)


def clean_sky_array(sky_array: numpy.ndarray) -> numpy.ndarray:
    """Remove sources with NaN or negative flux, and replace other NaNs."""
    # Remove sources with NaN in flux column.
    sky_array = sky_array[~numpy.isnan(sky_array[:, 2])]
    # Remove sources with negative fluxes (yes, there are about 3000 of those!).
    sky_array = sky_array[sky_array[:, 2] > 0]
    # Replace NaNs in spectral index column with zeros.
    return numpy.nan_to_num(sky_array)


def thin_out_sky_model(
    sky: oskar.Sky,
    bands,
    ra0_deg: float,
    dec0_deg: float,
    precision: str,
) -> oskar.Sky:
    """
    Keep a fraction of the brightest sources in each annulus around the
    phase centre.

    :param bands: List of (fraction, inner_radius_deg, outer_radius_deg).
    """
    sky_composite = None
    for fraction, inner_deg, outer_deg in bands:
        band = filter_sky_model(
            sky, fraction, inner_deg, outer_deg, ra0_deg, dec0_deg, precision
        )
        if sky_composite is None:
            sky_composite = band.create_copy()
        else:
            sky_composite.append(band)
    return sky_composite


def gleam_sky_model(
    path: str, ra0_deg: float, dec0_deg: float, max_radius_deg: float, precision: str
):
    """
    Generate a sky model from the GLEAM catalogue.

    :param path: GLEAM catalogue FITS file.
    :param ra0_deg: Central RA, in degrees.
    :param dec0_deg: Central Dec, in degrees.
    :param max_radius_deg: Maximum radius to keep.
    :param precision: OSKAR precision, "single" or "double".

    :return: The GLEAM-based sky model.
    """
    data = fits.getdata(path, 1)
    stokes_I = data["int_flux_143"]
    zeros = numpy.zeros_like(stokes_I)
    ref_freq_hz = numpy.ones_like(stokes_I) * 143e6
    sky_array = numpy.column_stack(
        (
            data["RAJ2000"],
            data["DEJ2000"],
            stokes_I,
            zeros,  # Q
            zeros,  # U
            zeros,  # V
            ref_freq_hz,
            data["alpha"],
        )
    )
    gleam = sky_from_array(clean_sky_array(sky_array), precision)
    gleam.filter_by_radius(0, max_radius_deg, ra0_deg, dec0_deg)
    # Keep everything inside 2.5 degrees, the brightest 50% of sources
    # between 2.5 and 5 degrees, and the brightest 10% outside 5 degrees.
    bands = [(1.0, 0.0, 2.5), (0.5, 2.5, 5.0), (0.1, 5.0, max_radius_deg)]
    return thin_out_sky_model(gleam, bands, ra0_deg, dec0_deg, precision)


def lotss_sky_model(
    path: str, ra0_deg: float, dec0_deg: float, max_radius_deg: float, precision: str
):
    """
    Generate a sky model from a LoTSS pointing.

    :param path: LoTSS DR2 catalogue FITS file.
    :param ra0_deg: Central RA, in degrees.
    :param dec0_deg: Central Dec, in degrees.
    :param max_radius_deg: Maximum radius to keep.
    :param precision: OSKAR precision, "single" or "double".

    :return: The LoTSS-based sky model.

    Note the DEC is negated from positive to negative so that the SKA-LOW array has an appropriate primary beam
    """
    data = fits.getdata(path, 1)
    stokes_I = data["Total_flux"] * 1e-3  # catalogue flux in units of mJy
    zeros = numpy.zeros_like(stokes_I)
    ref_freq_hz = numpy.ones_like(stokes_I) * 144e6
    # hard-coded spectral index until we get appropriate catalogue
    alpha = numpy.ones_like(stokes_I) * -0.7
    sky_array = numpy.column_stack(
        (
            data["RA"],
            data["DEC"] * -1.0,  # Negate dec to form appropriate beam in southern sky
            stokes_I,
            zeros,  # Q
            zeros,  # U
            zeros,  # V
            ref_freq_hz,
            alpha,
            zeros,  # RM: assume no polarized sources
            data["DC_Maj"],
            data["DC_Min"],
            data["DC_PA"],
        )
    )
    lotss = sky_from_array(clean_sky_array(sky_array).astype(numpy.float32), precision)
    lotss.filter_by_radius(0, max_radius_deg, ra0_deg, dec0_deg)
    # Keep everything inside 7.0 degrees, the brightest 50% of sources
    # between 7.0 and 8 degrees, and the brightest 10% outside 8 degrees.
    bands = [(1.0, 0.0, 7.0), (0.5, 7.01, 8.0), (0.1, 8.01, max_radius_deg)]
    return thin_out_sky_model(lotss, bands, ra0_deg, dec0_deg, precision)


def build_sky_model(
    run_cfg: SimulateVisibilityConfig, ra_deg: float, dec_deg: float, precision: str
):
    """Combine the sky models of the config file."""
    sky_composite = oskar.Sky(precision=precision)
    if run_cfg.gleam_file:
        print(f"Adding GLEAM sources from: {run_cfg.gleam_file}")
        sky_composite.append(
            gleam_sky_model(
                run_cfg.gleam_file,
                ra_deg,
                dec_deg,
                run_cfg.field_radius_deg,
                precision,
            )
        )
    if run_cfg.lotss_file:
        print(f"Adding LoTSS sources from: {run_cfg.lotss_file}")
        sky_composite.append(
            lotss_sky_model(
                run_cfg.lotss_file,
                ra_deg,
                dec_deg,
                run_cfg.field_radius_deg,
                precision,
            )
        )
    if run_cfg.sky_model_file:
        if not os.path.isfile(run_cfg.sky_model_file):
            raise RuntimeError(f"Sky model {run_cfg.sky_model_file} does not exist!")
        print(f"Adding sources from sky model file: {run_cfg.sky_model_file}")
        sky_composite.append(
            oskar.Sky.load(run_cfg.sky_model_file, precision=precision)
        )
    for name in run_cfg.bright_sources:
        print(f"Adding {name.replace('_', ' ').title()} to sky model")
        sky_composite.append(sky_from_array(BRIGHT_SOURCES[name], precision))
    print(f"Sky model contains {sky_composite.num_sources} sources.")
    return sky_composite


# ------------------------------- Inputs ------------------------------- #


def copy_file(source_file_path: str, output_dir_path: str):
    """
    Copy a file into output_dir_path, unless it is already the file there
    (e.g. the copy from a previous run).
    """
    try:
        shutil.copy(source_file_path, output_dir_path)
    except shutil.SameFileError:
        pass


def stage_inputs(config_path: str, output_dir: str, run_cfg: SimulateVisibilityConfig):
    """
    Copy the config file and the custom sky model into "inputs" in the output
    directory, and link the TEC screen (which can be large) there, so that the
    outputs keep a record of what was simulated.
    """
    inputs_dir = os.path.join(output_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    copy_file(config_path, inputs_dir)
    if run_cfg.sky_model_file:
        copy_file(run_cfg.sky_model_file, inputs_dir)
    if run_cfg.tec_screen:
        link = os.path.join(inputs_dir, Path(run_cfg.tec_screen).name)
        if os.path.lexists(link):
            # From a previous run in the same output directory
            os.remove(link)
        os.symlink(run_cfg.tec_screen, link)


# --------------------------- Telescope model --------------------------- #


def stage_tel_model(
    tel_model: str, tel_model_copy: str, run_cfg: SimulateVisibilityConfig
):
    """
    Copy the telescope model, and link the gain table and cable delays of the
    config file into the copy, with the names expected by OSKAR. Corruption
    files of the model itself are not copied, so that corruptions are only
    applied when set in the config file.
    """
    if os.path.isdir(tel_model_copy):
        # From a previous run in the same output directory
        shutil.rmtree(tel_model_copy)
    skipped = []
    for key, name in TEL_MODEL_CORRUPTION_FILES.items():
        if os.path.lexists(os.path.join(tel_model, name)):
            print(
                f"WARNING: not copying {name} of the telescope model "
                f"{tel_model}. Set simulate_visibility.{key} to apply it."
            )
            skipped.append(name)
    shutil.copytree(
        tel_model,
        tel_model_copy,
        ignore=lambda directory, names: skipped if directory == tel_model else [],
    )
    for key, name in TEL_MODEL_CORRUPTION_FILES.items():
        path = getattr(run_cfg, key)
        if path:
            print(f"Linking {path} into the telescope model as {name}")
            os.symlink(path, os.path.join(tel_model_copy, name))


# ------------------------------ Settings ------------------------------ #


def interferometer_settings(
    args,
    cfg: SimConfig,
    run_cfg: SimulateVisibilityConfig,
    tel_model: str,
    target: Target,
    start_time,
    scan_length_mins,
    num_times,
    num_channels,
    ms_path,
) -> dict:
    """Settings for oskar_sim_interferometer."""
    sim_params = {
        "simulator/double_precision": run_cfg.double_precision,
        "simulator/use_gpus": run_cfg.use_gpus,
        "simulator/max_sources_per_chunk": args.max_sources_per_chunk,
        "simulator/keep_log_file": True,
        "simulator/write_status_to_log_file": True,
        "observation/phase_centre_ra_deg": target.ra_deg,
        "observation/phase_centre_dec_deg": target.dec_deg,
        "observation/start_frequency_hz": cfg.simulation_start_frequency_hz,
        "observation/num_channels": num_channels,
        "observation/frequency_inc_hz": cfg.correlated_channel_bandwidth_hz,
        "observation/start_time_utc": str(start_time),
        "observation/length": 60 * scan_length_mins,
        "observation/num_time_steps": num_times,
        "telescope/input_directory": tel_model,
        "telescope/station_type": run_cfg.station_type,
        "telescope/normalise_beams_at_phase_centre": False,
        "telescope/aperture_array/array_pattern/normalise": True,
        "interferometer/channel_bandwidth_hz": cfg.correlated_channel_bandwidth_hz,
        "interferometer/time_average_sec": cfg.sampling_time_sec,
        "interferometer/max_time_samples_per_block": 8,
        "interferometer/max_channels_per_block": num_channels,
        "interferometer/ms_filename": ms_path,
        "interferometer/ms_dish_diameter": 38,
        "interferometer/noise/enable": run_cfg.add_noise,
        "interferometer/noise/freq": "Range",
        "interferometer/noise/freq/start": cfg.simulation_start_frequency_hz,
        "interferometer/noise/freq/inc": cfg.correlated_channel_bandwidth_hz,
        "interferometer/noise/freq/number": num_channels,
        "interferometer/noise/rms": "Range",
        "interferometer/noise/rms/start": 0.005,
        "interferometer/noise/rms/end": 0.01,
    }

    taper_key = "telescope/aperture_array/element_pattern/taper"
    if run_cfg.element_taper != "None":
        sim_params[f"{taper_key}/type"] = run_cfg.element_taper
    if run_cfg.element_taper == "Cosine":
        sim_params[f"{taper_key}/cosine_power"] = run_cfg.element_cosine_power
    if run_cfg.element_taper == "Gaussian":
        sim_params[f"{taper_key}/gaussian_fwhm_deg"] = run_cfg.element_gaussian_fwhm_deg

    # OSKAR does not simulate smearing if these are 0 (their defaults).
    # The MS keeps the channel width (from the frequency increment), but
    # EXPOSURE is written as 0.
    if run_cfg.no_smearing:
        sim_params["interferometer/channel_bandwidth_hz"] = 0.0
        sim_params["interferometer/time_average_sec"] = 0.0

    if run_cfg.tec_screen:
        if not os.path.isfile(run_cfg.tec_screen):
            raise RuntimeError(f"TEC screen {run_cfg.tec_screen} does not exist!")
        print(f"Using TEC screen (exists): {run_cfg.tec_screen}")
        sim_params["telescope/ionosphere_screen_type"] = "External"
        sim_params["telescope/external_tec_screen/input_fits_file"] = run_cfg.tec_screen

    return sim_params


def run_imager(imager_cfg: OskarImagerConfig, ms_path: str, psf: bool = False):
    """Make a dirty image (or the PSF) of the MS with the OSKAR imager."""
    img_params = {
        "image/size": imager_cfg.image_size,
        "image/specify_cellsize": True,
        "image/cellsize_arcsec": imager_cfg.cellsize_arcsec,
        "image/algorithm": "W-projection",
        "image/fft/use_gpu": True,
        "image/fft/grid_on_gpu": True,
        "image/input_vis_data": ms_path,
        "image/root_path": os.path.splitext(ms_path)[0],
    }
    if psf:
        img_params["image/image_type"] = "PSF"
    settings = oskar.SettingsTree("oskar_imager")
    settings.from_dict(img_params)
    oskar.Imager(settings=settings).run()


def main():
    """
    Main function to run simulations.
    """
    args = parse_args()

    # Print arguments.
    print("Script argument list:")
    print("\n".join(f"    {k}={v}" for k, v in vars(args).items()))

    cfg = SimConfig.load(args.config)
    # Resolve the relative paths before changing directory
    config_path = os.path.realpath(args.config)
    tel_model_name = Path(
        os.path.abspath(cfg.require_simulate_visibility().tel_model)
    ).name
    run_cfg = cfg.require_simulate_visibility().with_resolved_paths()
    tel_model = run_cfg.tel_model
    output_dir = os.path.abspath(args.output_dir)
    # Named as in the config file, in case the telescope model is a link
    tel_model_copy = os.path.join(output_dir, tel_model_name + ".custom")

    if args.field and args.target:
        field_name, target_name = args.field, args.target
        target = cfg.get_target(field_name, target_name)
    elif args.field or args.target:
        raise RuntimeError("--field and --target must be given together")
    else:
        field_name, target_name, target = cfg.first_target()
    print(f"Target: {field_name} / {target_name}")

    # Find telescope model name to use, and check that it exists.
    if not os.path.isdir(tel_model):
        raise RuntimeError(f"Telescope model {tel_model} does not exist!")

    # Set up the scan.
    obs_length_mins = cfg.observing_time_mins
    start_time = target.scan_start_time(
        obs_length_mins, args.scan_index, args.num_scans
    )
    scan_length_mins = obs_length_mins / args.num_scans
    num_times = int(numpy.floor(60 * scan_length_mins / cfg.sampling_time_sec))
    num_channels = int(
        numpy.floor(
            (cfg.simulation_end_frequency_hz - cfg.simulation_start_frequency_hz)
            / cfg.correlated_channel_bandwidth_hz
        )
    )
    ms_path = os.path.join(
        output_dir,
        MS_NAME_FORMAT.format(scan_number=target.scan_number(args.scan_index)),
    )
    print(f"Running scan {args.scan_index + 1} / {args.num_scans}")
    print(f"Scan start time: {str(start_time)}")
    print(f"Scan length (minutes): {scan_length_mins}")
    print(f"Using telescope model (exists): {tel_model}")
    print(f"Telescope model copy, used by OSKAR: {tel_model_copy}")
    print(f"Number of time samples in scan: {num_times}")
    print(f"Number of channels: {num_channels}")
    print(f"Output Measurement Set: {ms_path}")

    if run_cfg.double_precision:
        print("Using double precision for oskar computations.")
        precision = "double"
    else:
        print("Using single precision for oskar computations.")
        precision = "single"

    sim_params = interferometer_settings(
        args,
        cfg,
        run_cfg,
        tel_model_copy,
        target,
        start_time,
        scan_length_mins,
        num_times,
        num_channels,
        ms_path,
    )
    sky_composite = build_sky_model(run_cfg, target.ra_deg, target.dec_deg, precision)

    # Stop now if doing a dry run.
    if args.dry_run:
        print("Dry run specified: Exiting.")
        return

    # Ensure the output directory exists, and run from it, so that the files
    # written by OSKAR in the current directory end up there.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    stage_inputs(config_path, output_dir, run_cfg)
    stage_tel_model(tel_model, tel_model_copy, run_cfg)

    # Create the sky model and region file if necessary.
    sky_path_csv = os.path.join(output_dir, "sky_model.csv")
    sky_path_reg = os.path.join(output_dir, "sky_model.reg")
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

    if run_cfg.oskar_imager.make_image:
        run_imager(run_cfg.oskar_imager, ms_path)
    if run_cfg.oskar_imager.make_psf:
        run_imager(run_cfg.oskar_imager, ms_path, psf=True)


if __name__ == "__main__":
    main()
