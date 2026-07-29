import logging
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from resources.constants import (  # pylint: disable=import-error
    CABLE_DELAYS,
    OUTLIER_CHANNEL_INDICES,
    OUTLIER_STATION_INDICES,
)
from resources.data_sim import (  # pylint: disable=import-error
    generate_calibrator_data,
    init_cal_config,
    migrate_sky_model,
)

logger = logging.getLogger("INST INTEGRATION")
logging.basicConfig(level=logging.INFO)


def read_h5parm_gains(h5parm_path):
    """Read pols, frequencies and first-time amplitude/phase values
    from a H5Parm file.
    """
    with h5py.File(h5parm_path) as f:
        pols = [
            p.decode("ascii").rstrip("\x00")
            for p in f["sol000/amplitude000/pol"][:]
        ]
        freq = f["sol000/amplitude000/freq"][:]
        amp = f["sol000/amplitude000/val"][0]  # (ant, freq, pol)
        phase = f["sol000/phase000/val"][0]  # (ant, freq, pol)

    return pols, freq, amp, phase


def validate_inst_gaintable(output_dir, temp_path, field_id, refant=0):
    expected_gaintable_path = temp_path / "sim_gaintable.h5parm"
    actual_gaintable_path = output_dir / f"{field_id}_inst.gaintable.h5parm"

    expected_pols, expected_freq, expected_amp, expected_phase = (
        read_h5parm_gains(expected_gaintable_path)
    )
    actual_pols, actual_freq, actual_amp, actual_phase = read_h5parm_gains(
        actual_gaintable_path
    )

    np.testing.assert_allclose(
        actual_freq,
        expected_freq,
        err_msg="Frequencies don't match between INST and simulated gaintable",
    )

    n_stations, n_channels, _ = actual_amp.shape
    expected_flagged = np.zeros((n_stations, n_channels), dtype=bool)
    expected_flagged[
        np.ix_(OUTLIER_STATION_INDICES, OUTLIER_CHANNEL_INDICES)
    ] = True

    for pol_name in ("XX", "YY"):
        expected_pol_idx = expected_pols.index(pol_name)
        actual_pol_idx = actual_pols.index(pol_name)

        expected_amp_pol = expected_amp[:, :, expected_pol_idx]
        expected_phase_pol = expected_phase[:, :, expected_pol_idx]
        expected_phase_pol = (
            expected_phase_pol - expected_phase_pol[[refant], :]
        )

        actual_amp_pol = actual_amp[:, :, actual_pol_idx]
        actual_phase_pol = actual_phase[:, :, actual_pol_idx]
        actual_phase_pol = actual_phase_pol - actual_phase_pol[[refant], :]

        np.testing.assert_array_equal(
            np.isnan(actual_amp_pol),
            expected_flagged,
            err_msg=f"Flagged gains for {pol_name} do not match",
        )

        np.testing.assert_allclose(
            actual_amp_pol[~expected_flagged],
            expected_amp_pol[~expected_flagged],
            rtol=0.05,
            atol=0.05,
            err_msg=f"{pol_name} amplitudes do not "
            f"match the between INST and simulated gaintables",
        )

        phase_diff = np.angle(
            np.exp(
                1j
                * (
                    actual_phase_pol[~expected_flagged]
                    - expected_phase_pol[~expected_flagged]
                )
            )
        )
        np.testing.assert_allclose(
            phase_diff,
            0,
            atol=np.deg2rad(5),
            err_msg=f"{pol_name} phases do not "
            f"match between INST and simulated gaintables",
        )


def validate_delay_stage(output_dir, ms_path, refant=0):
    speed_of_light = 299792458.0

    cable_length_errors = np.loadtxt(CABLE_DELAYS, dtype=np.float64)
    expected_delays = cable_length_errors / speed_of_light
    expected_delays = expected_delays - expected_delays[refant]

    ms_prefix = Path(ms_path).resolve().stem
    actual_delaytable_path = os.path.join(
        output_dir, f"gaintables/{ms_prefix}/delay.clock.h5parm"
    )

    with h5py.File(actual_delaytable_path) as f:
        pols = [
            p.decode("ascii").rstrip("\x00")
            for p in f["sol000/clock000/pol"][:]
        ]
        actual_delay = f["sol000/clock000/val"][0]  # (ant, pol)

    actual_delay = actual_delay[:, [pols.index("XX"), pols.index("YY")]]
    actual_delay = actual_delay - actual_delay[refant]

    for pol_idx, pol_name in enumerate(("XX", "YY")):
        np.testing.assert_allclose(
            actual_delay[:, pol_idx],
            expected_delays,
            rtol=1e-3,
            atol=1e-9,
            err_msg=f"{pol_name} delays do not match the expected delays",
        )


class IntegrationTest(unittest.TestCase):
    def test_instrumental_calibration_integration(self):
        """Run integration test for Instrumental claibration Pipeline"""
        logger.info(
            "Run integration test for Instrumental claibration Pipeline"
        )
        field_id = "CAL_FIELD"
        scan_intent = "CALIBRATE_BANDPASS#ON_SOURCE"
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            logger.info(
                "Temp folder: %s. Running test in %s", tmpdirname, os.getcwd()
            )
            temp_path = Path(tmpdirname)
            input_ms_path = generate_calibrator_data(
                temp_path, field_id, scan_intent, corrupt=True
            )

            lsm_path = migrate_sky_model(temp_path)
            inst_config_path = init_cal_config(
                temp_path, ms_path=input_ms_path, lsm_path=lsm_path
            )
            output_dir = temp_path / "inst_output"

            command = [
                "ska-sdp-instrumental-calibration",
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

            validate_delay_stage(output_dir, input_ms_path)
            validate_inst_gaintable(output_dir, temp_path, field_id)


if __name__ == "__main__":
    unittest.main()
