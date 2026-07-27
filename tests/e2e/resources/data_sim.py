import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from string import Template

import h5py
import numpy
import oskar  # pylint: disable=import-error
import pandas as pd
from casacore.tables import table
from ska_sdp_datamodels.global_sky_model import LocalSkyModel

from .constants import (
    CABLE_DELAYS,
    CHANNEL_WIDTH_HZ,
    END_FREQ_HZ,
    INST_CAL_CONFIG,
    N_STATIONS,
    OBSERVING_TIME_MINS,
    OUTLIER_AMPLITUDE,
    OUTLIER_CHANNEL_INDICES,
    OUTLIER_PHASE_DEG,
    OUTLIER_STATION_INDICES,
    SAMPLING_TIME_SEC,
    SKY_MODEL,
    START_FREQ_HZ,
    TEL_MODEL,
)
from .generate_gaintable import calculate_gains

logger = logging.getLogger("DATA-SIM")


def add_field_id_and_scan_intent_to_source(ms_path, field_id, scan_intent):
    """
    Updates the source measurement set paths to have SCAN and FIELD id
    """
    logger.info("Add field_id %s and scan_intent %s", field_id, scan_intent)
    if scan_intent is not None:
        with table(os.path.join(ms_path, "STATE"), readonly=False) as ms:
            ms.putcell("OBS_MODE", 0, scan_intent)

    if field_id is not None:
        with table(os.path.join(ms_path, "FIELD"), readonly=False) as ms:
            ms.putcell("NAME", 0, field_id)


def init_cal_config(
    output_dir,
    *,
    ms_path,
    lsm_path,
):
    """Initialise CAL configuration"""
    config_path = output_dir / "inst_config.yaml"
    with open(INST_CAL_CONFIG, mode="r", encoding="utf-8") as config_template:
        template = Template(config_template.read())
        inst_config = template.safe_substitute(
            SKY_MODEL_PATH=lsm_path, MS_PATH=ms_path
        )

        with open(config_path, mode="w", encoding="utf-8") as config_file:
            config_file.write(inst_config)

    return config_path


def migrate_sky_model(output_dir):
    """Migrate from oskar csv to ska-lsm format"""
    df = pd.read_csv(
        SKY_MODEL,
        header=None,
        comment="#",
        names=[
            "ra_deg",
            "dec_deg",
            "i_pol_jy",
            "Q_",
            "U_",
            "V_",
            "ref_freq_hz",
            "spec_idx",
            "R_",
            "a_arcsec",
            "b_arcsec",
            "pa_deg",
        ],
    )
    df["component_id"] = "OSKAR " + df.index.astype(str).str.zfill(6)
    df["spec_idx"] = df["spec_idx"].apply(
        lambda x: [x] if pd.notna(x) else [0.0]
    )
    df["log_spec_idx"] = True
    df["a_arcsec"] = df["a_arcsec"] / 2
    df["b_arcsec"] = df["b_arcsec"] / 2
    df["source_id"] = ""
    df["epoch"] = 0.0
    df = df[
        [
            "component_id",
            "source_id",
            "ra_deg",
            "dec_deg",
            "i_pol_jy",
            "ref_freq_hz",
            "epoch",
            "a_arcsec",
            "b_arcsec",
            "pa_deg",
            "spec_idx",
            "log_spec_idx",
        ]
    ]

    column_names = [col for col in df]
    vector_columns = ["spec_idx"]

    local_sky_model = LocalSkyModel(
        column_names=column_names,
        num_rows=len(df),
        vector_columns=vector_columns,
    )

    for idx, row in enumerate(df.to_dict("records")):
        local_sky_model.set_row(idx, row)

    lsm_path = os.path.join(output_dir, "local_sky_model.csv")
    local_sky_model.save(lsm_path)

    return lsm_path


def generate_calibrator_data(output_dir, field_id, scan_intent, corrupt=False):
    """Generate calibrator data"""

    transit_time = datetime.fromisoformat("2000-01-03 22:33:30.000")
    start_time = transit_time - timedelta(minutes=0.5)
    num_times = int(numpy.floor(60 / SAMPLING_TIME_SEC))

    num_channels = int(
        numpy.floor((END_FREQ_HZ - START_FREQ_HZ) / CHANNEL_WIDTH_HZ)
    )

    ms_name = "corrupted_visibility.ms" if corrupt else "test_visbility.ms"
    ms_path = os.path.join(output_dir, ms_name)

    tel_model = TEL_MODEL
    corrupted_tel_model = None
    if corrupt:
        gaintable_path = generate_bandpass_gaintable(output_dir)
        corrupted_tel_model = Path(output_dir) / "corrupted_tel_model.tm"
        shutil.copytree(TEL_MODEL, corrupted_tel_model)
        (corrupted_tel_model / "gain_model.h5").symlink_to(
            Path(gaintable_path).resolve()
        )
        (corrupted_tel_model / "cable_length_error_18s.txt").symlink_to(
            Path(CABLE_DELAYS).resolve()
        )
        tel_model = corrupted_tel_model

    sim_params = {
        "simulator/double_precision": True,
        "simulator/use_gpus": False,
        "simulator/max_sources_per_chunk": 2000,
        "simulator/keep_log_file": True,
        "simulator/write_status_to_log_file": True,
        "observation/phase_centre_ra_deg": 197.914612,
        "observation/phase_centre_dec_deg": -22.277973,
        "observation/start_frequency_hz": START_FREQ_HZ,
        "observation/num_channels": num_channels,
        "observation/frequency_inc_hz": CHANNEL_WIDTH_HZ,
        "observation/start_time_utc": str(start_time),
        "observation/length": 60,
        "observation/num_time_steps": num_times,
        "telescope/input_directory": tel_model,
        "telescope/normalise_beams_at_phase_centre": False,
        "telescope/aperture_array/array_pattern/normalise": True,
        "interferometer/channel_bandwidth_hz": CHANNEL_WIDTH_HZ,
        "interferometer/time_average_sec": SAMPLING_TIME_SEC,
        "interferometer/max_time_samples_per_block": 8,
        "interferometer/max_channels_per_block": num_channels,
        "interferometer/ms_filename": ms_path,
        "interferometer/ms_dish_diameter": 38,
    }

    logger.info(
        "Starting data generation for integration test with parameters %s.",
        sim_params,
    )
    logger.info("Data generation will take some time")
    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(sim_params)
    sim = oskar.Interferometer(settings=settings)

    sky = oskar.Sky()
    sky_composite = sky.load(SKY_MODEL)
    sim.set_sky_model(sky_composite)

    sim.run()
    add_field_id_and_scan_intent_to_source(ms_path, field_id, scan_intent)
    logger.info("Finished data generation, MS path: %s", ms_path)

    if corrupted_tel_model is not None:
        shutil.rmtree(corrupted_tel_model)

    return ms_path


def generate_bandpass_gaintable(output_dir):
    """Generate a synthetic bandpass gaintable"""

    gain_xpol, gain_ypol, sim_freqs = calculate_gains(
        n_stations=N_STATIONS,
        start_frequency_hz=START_FREQ_HZ,
        end_frequency_hz=END_FREQ_HZ,
        channel_bandwidth_hz=CHANNEL_WIDTH_HZ,
        sampling_time_sec=SAMPLING_TIME_SEC,
        observing_time_mins=OBSERVING_TIME_MINS,
        outlier_station_indices=OUTLIER_STATION_INDICES,
        outlier_channel_indices=OUTLIER_CHANNEL_INDICES,
        outlier_amplitude=OUTLIER_AMPLITUDE,
        outlier_phase_deg=OUTLIER_PHASE_DEG,
    )

    gaintable_path = os.path.join(output_dir, "sim_gaintable.h5")
    with h5py.File(gaintable_path, "w") as f:
        f.create_dataset("freq (Hz)", data=sim_freqs * 1e6)
        f.create_dataset("gain_xpol", data=gain_xpol)
        f.create_dataset("gain_ypol", data=gain_ypol)

    logger.info("Finished gaintable generation, path: %s", gaintable_path)

    return gaintable_path
