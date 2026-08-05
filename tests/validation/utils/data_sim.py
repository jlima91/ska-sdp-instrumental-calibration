import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from string import Template

import numpy
import oskar  # pylint: disable=import-error
import pandas as pd
from casacore.tables import table
from resources import (
    CABLE_DELAYS,
    INST_CAL_CONFIG,
    INST_TARGET_CONFIG,
    SKY_MODEL,
    TEL_MODEL,
)
from ska_sdp_datamodels.global_sky_model import LocalSkyModel

from .constants import (
    CHANNEL_WIDTH_HZ,
    END_FREQ_HZ,
    EOR2_CAL_DEC,
    EOR2_CAL_RA,
    SAMPLING_TIME_SEC,
    START_FREQ_HZ,
    TRANSIT_TIME,
)
from .generate_gaintable import (
    generate_bandpass_gaintable,
    generate_target_gaintable,
)

logger = logging.getLogger("DATA-SIM")


def add_gain_corruptions(ms_path, h5parm_path):
    """Corrupt the MS visibilities in place with the bandpass gains from
    the H5Parm, using DP3.
    """
    command = [
        "DP3",
        f"msin={ms_path}",
        "msout=.",
        "msout.datacolumn=DATA",
        "steps=[applycal]",
        "applycal.type=applycal",
        f"applycal.parmdb={h5parm_path}",
        "applycal.steps=[amplitude,phase]",
        "applycal.amplitude.correction=amplitude000",
        "applycal.phase.correction=phase000",
        "applycal.invert=false",
    ]
    logger.info("Applying gain corruptions: %s", command)
    subprocess.run(command, check=True)


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


def init_target_config(
    output_dir,
    *,
    ms_path,
    lsm_path,
):
    """Initialise CAL configuration"""
    config_path = output_dir / "inst_config.yaml"
    with open(
        INST_TARGET_CONFIG, mode="r", encoding="utf-8"
    ) as config_template:
        template = Template(config_template.read())
        inst_config = template.safe_substitute(
            SKY_MODEL_PATH=lsm_path, MS_PATH=ms_path
        )

        with open(config_path, mode="w", encoding="utf-8") as config_file:
            config_file.write(inst_config)

    return config_path


def migrate_sky_model(sky_model_path, output_dir):
    """Migrate from oskar csv to ska-lsm format"""
    df = pd.read_csv(
        sky_model_path,
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

    transit_time = datetime.fromisoformat(TRANSIT_TIME)
    start_time = transit_time - timedelta(minutes=0.5)
    num_times = int(numpy.floor(60 / SAMPLING_TIME_SEC))

    num_channels = int(
        numpy.floor((END_FREQ_HZ - START_FREQ_HZ) / CHANNEL_WIDTH_HZ)
    )

    ms_name = "corrupted_visibility.ms" if corrupt else "test_visibility.ms"
    ms_path = os.path.join(output_dir, ms_name)

    tel_model = TEL_MODEL
    corrupted_tel_model = None

    if corrupt:
        corrupted_tel_model = Path(output_dir) / "corrupted_tel_model.tm"

        shutil.copytree(TEL_MODEL, corrupted_tel_model)
        (corrupted_tel_model / "cable_length_error.txt").symlink_to(
            Path(CABLE_DELAYS).resolve()
        )

        tel_model = corrupted_tel_model

    sim_params = {
        "simulator/double_precision": True,
        "simulator/use_gpus": False,
        "simulator/max_sources_per_chunk": 2000,
        "simulator/keep_log_file": True,
        "simulator/write_status_to_log_file": True,
        "observation/phase_centre_ra_deg": EOR2_CAL_RA,
        "observation/phase_centre_dec_deg": EOR2_CAL_DEC,
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

    if corrupt:
        h5parm_path = generate_bandpass_gaintable(output_dir, start_time)
        add_gain_corruptions(ms_path, h5parm_path)

    logger.info("Finished data generation, MS path: %s", ms_path)

    if corrupted_tel_model is not None:
        shutil.rmtree(corrupted_tel_model)

    return ms_path


def generate_target_data(output_dir, field_id, scan_intent, corrupt=False):
    """Generate calibrator data"""

    transit_time = datetime.fromisoformat(TRANSIT_TIME)
    start_time = transit_time - timedelta(minutes=0.5)
    num_times = int(numpy.floor(60 / SAMPLING_TIME_SEC))

    num_channels = int(
        numpy.floor((END_FREQ_HZ - START_FREQ_HZ) / CHANNEL_WIDTH_HZ)
    )

    ms_name = "corrupted_visibility.ms" if corrupt else "test_visibility.ms"
    ms_path = os.path.join(output_dir, ms_name)

    tel_model = TEL_MODEL
    corrupted_tel_model = None

    sim_params = {
        "simulator/double_precision": True,
        "simulator/use_gpus": False,
        "simulator/max_sources_per_chunk": 2000,
        "simulator/keep_log_file": True,
        "simulator/write_status_to_log_file": True,
        "observation/phase_centre_ra_deg": EOR2_CAL_RA,
        "observation/phase_centre_dec_deg": EOR2_CAL_DEC,
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

    if corrupt:
        h5parm_path = generate_target_gaintable(output_dir, start_time)
        add_gain_corruptions(ms_path, h5parm_path)

    logger.info("Finished data generation, MS path: %s", ms_path)

    if corrupted_tel_model is not None:
        shutil.rmtree(corrupted_tel_model)

    return ms_path
