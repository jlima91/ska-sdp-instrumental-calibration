import logging
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from resources import (  # pylint: disable=import-error
    INST_TARGET_COMPLEX_GAIN_CONFIG,
    INST_TARGET_IONOSPHERIC_CONFIG,
    SKY_MODEL,
)
from utils.constants import RANDOM_SEED  # pylint: disable=import-error
from utils.data_sim import (  # pylint: disable=import-error
    apply_gain_corrections,
    generate_target_data,
    init_config,
    migrate_sky_model,
)

from ska_sdp_instrumental_calibration.data_managers.visibility import (
    load_ms_as_dataset_with_time_chunks,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logger = logging.getLogger("INST INTEGRATION")
logging.basicConfig(level=logging.INFO)


VALIDATION_STATIONS = [2, 8, 17]


def read_h5parm_gains(h5parm_path):
    """Read pols, frequencies and first-time amplitude/phase values
    from a H5Parm file.
    """
    with h5py.File(h5parm_path) as f:
        pols = [
            p.decode("ascii").rstrip("\x00")
            for p in f["sol000/amplitude000/pol"][:]
        ]
        time = f["sol000/amplitude000/time"][:]
        freq = f["sol000/amplitude000/freq"][:]
        phase = f["sol000/phase000/val"][:, :, :, :]  # (time, ant, freq, pol)

    return pols, time, freq, phase


def reference_phase_to_refant(phase, refant):
    """Remove the phase gauge freedom by zeroing the refant phase"""
    return phase - phase[:, [refant]]


def normalised_rmse(actual, expected):
    """RMSE of (actual - expected), as a fraction of the band mean"""
    return np.sqrt(np.mean((actual - expected) ** 2)) / np.mean(expected)


def wrapped_phase_rmse(actual, expected):
    """RMSE of phase differences across one station's channels, wrapped
    to (-pi, pi]"""
    phase_error = np.angle(np.exp(1j * (actual - expected)))
    return np.sqrt(np.mean(phase_error**2))


def validate_complex_gaintable(output_dir, temp_path, field_id, refant=0):
    phase_rmse_threshold = np.deg2rad(6)

    expected_pols, expected_time, _, expected_phase = read_h5parm_gains(
        temp_path / "sim_gaintable.h5parm"
    )
    actual_pols, actual_time, _, actual_phase = read_h5parm_gains(
        output_dir / f"{field_id}_inst.gaintable.h5parm"
    )

    np.testing.assert_allclose(
        actual_time,
        expected_time,
        err_msg="Times don't match between INST and simulated gaintable",
    )

    for pol_name in ("XX", "YY"):
        expected_pol_idx = expected_pols.index(pol_name)
        actual_pol_idx = actual_pols.index(pol_name)

        expected_phase_pol = reference_phase_to_refant(
            expected_phase[:, :, 0, expected_pol_idx], refant
        )

        actual_phase_pol = reference_phase_to_refant(
            actual_phase[:, :, 0, actual_pol_idx], refant
        )

        for station in VALIDATION_STATIONS:
            phase_rmse = wrapped_phase_rmse(
                actual_phase_pol[:, station], expected_phase_pol[:, station]
            )

            assert phase_rmse < phase_rmse_threshold, (
                f"{pol_name} station {station} phase RMSE "
                f"{np.rad2deg(phase_rmse):.2f} deg exceeds threshold "
                f"{np.rad2deg(phase_rmse_threshold):.2f} deg"
            )


def validate_ionospheric(input_data_path, gaintable):

    phase_freq_threshold = 7

    phase_time_threshold = 5

    corrected_vis = load_ms_as_dataset_with_time_chunks(
        input_data_path, 30, datacolumn="CORRECTED_DATA"
    ).load()

    expected_time = corrected_vis.vis.time

    expected_freq = corrected_vis.vis.frequency

    expected_pols = ["XX", "YY"]

    actual_pols, actual_time, actual_freq, _ = read_h5parm_gains(gaintable)

    np.testing.assert_allclose(
        actual_time,
        expected_time,
        err_msg="Times don't match between INST gaintable and visibility",
    )

    np.testing.assert_allclose(
        actual_freq,
        expected_freq,
        err_msg="Freq don't match between INST gaintable and visibility",
    )

    assert (
        actual_pols == expected_pols
    ), "Gaintable pols don't match expected pols"

    vis = corrected_vis.vis.isel(
        baselineid=VALIDATION_STATIONS,
        polarisation=[0, 3],
    )

    mean_phase_freq = np.max(
        np.angle(
            vis.mean(dim=["time", "frequency"]),
            deg=True,
        )
    )

    np.testing.assert_allclose(
        mean_phase_freq,
        0,
        atol=phase_freq_threshold,
        err_msg=(
            f"Mean phase across frequency is not close to zero "
            f"Max Mean phase = {mean_phase_freq:.3f} deg, "
            f"threshold = ±{phase_freq_threshold:.3f} deg."
        ),
    )

    vis_phase_time = np.angle(
        vis.mean(dim="frequency"),
        deg=True,
    )

    phase_time_diff = np.max(np.diff(vis_phase_time, axis=0))

    np.testing.assert_allclose(
        phase_time_diff,
        0,
        atol=phase_time_threshold,
        err_msg=(
            f"Maximum phase difference = "
            f"Max Mean time diff = {phase_time_diff:.3f} deg "
            f"(threshold ±{phase_time_threshold:.3f} deg)."
        ),
    )


class TargetCalibration(unittest.TestCase):

    def test_target_complex_gain_calibration(self):
        """Run integration test for Instrumental calibration Pipeline"""
        logger.info(
            "Run integration test for Instrumental calibration Pipeline"
        )
        field_id = "TARGET_FIELD"
        scan_intent = "CALIBRATE_BANDPASS#ON_SOURCE"
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            logger.info(
                "Temp folder: %s. Running test in %s", tmpdirname, os.getcwd()
            )
            temp_path = Path(tmpdirname)
            input_ms_path = generate_target_data(
                temp_path, field_id, scan_intent, corrupt=True
            )

            lsm_path = migrate_sky_model(SKY_MODEL, temp_path)
            inst_config_path = init_config(
                INST_TARGET_COMPLEX_GAIN_CONFIG,
                temp_path,
                ms_path=input_ms_path,
                lsm_path=lsm_path,
            )
            output_dir = temp_path / "inst_output"

            command = [
                "ska-sdp-instrumental-target-calibration",
                "run",
                input_ms_path,
                "--config",
                inst_config_path,
                "--output",
                output_dir,
                "--no-unique-output-subdir",
            ]

            logger.info("Running command %s", command)
            subprocess.run(
                command,
                check=True,
            )

            validate_complex_gaintable(output_dir, temp_path, field_id)

    def test_target_ionospheric_calibration(self):
        """Run integration test for Instrumental calibration Pipeline"""
        logger.info(
            "Run integration test for Instrumental calibration Pipeline"
        )
        field_id = "TARGET_FIELD"
        scan_intent = "CALIBRATE_BANDPASS#ON_SOURCE"
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            logger.info(
                "Temp folder: %s. Running test in %s", tmpdirname, os.getcwd()
            )
            temp_path = Path(tmpdirname)
            input_ms_path = generate_target_data(
                temp_path, field_id, scan_intent, tec_screen=True
            )

            lsm_path = migrate_sky_model(SKY_MODEL, temp_path)
            inst_config_path = init_config(
                INST_TARGET_IONOSPHERIC_CONFIG,
                temp_path,
                ms_path=input_ms_path,
                lsm_path=lsm_path,
            )
            output_dir = temp_path / "inst_output"

            command = [
                "ska-sdp-instrumental-target-ionospheric",
                "run",
                input_ms_path,
                "--config",
                inst_config_path,
                "--output",
                output_dir,
                "--no-unique-output-subdir",
            ]

            logger.info("Running command %s", command)
            subprocess.run(
                command,
                check=True,
            )

            target_gain_table = (
                output_dir / f"{field_id}_inst.gaintable.h5parm"
            )

            apply_gain_corrections(input_ms_path, target_gain_table)

            validate_ionospheric(input_ms_path, target_gain_table)


if __name__ == "__main__":
    unittest.main()
