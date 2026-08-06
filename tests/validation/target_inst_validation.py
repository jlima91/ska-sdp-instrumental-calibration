import logging
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from resources import SKY_MODEL  # pylint: disable=import-error
from utils.data_sim import (  # pylint: disable=import-error
    generate_target_data,
    init_target_config,
    migrate_sky_model,
)

logger = logging.getLogger("INST INTEGRATION")
logging.basicConfig(level=logging.INFO)

VALIDATION_STATIONS = [1, 8, 17]


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
        phase = f["sol000/phase000/val"][:, :, 0, :]  # (time, ant, freq, pol)

    return pols, time, phase


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


def validate_inst_gaintable(output_dir, temp_path, field_id, refant=0):
    phase_rmse_threshold = np.deg2rad(2)

    expected_pols, expected_time, expected_phase = read_h5parm_gains(
        temp_path / "sim_gaintable.h5parm"
    )
    actual_pols, actual_time, actual_phase = read_h5parm_gains(
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
            expected_phase[:, :, expected_pol_idx], refant
        )

        actual_phase_pol = reference_phase_to_refant(
            actual_phase[:, :, actual_pol_idx], refant
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
            inst_config_path = init_target_config(
                temp_path, ms_path=input_ms_path, lsm_path=lsm_path
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

            validate_inst_gaintable(output_dir, temp_path, field_id)


if __name__ == "__main__":
    unittest.main()
