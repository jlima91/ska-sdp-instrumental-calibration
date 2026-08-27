import logging
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from resources import (  # pylint: disable=import-error
    CABLE_DELAYS,
    INST_CAL_CONFIG,
    SKY_MODEL,
)
from utils.constants import (  # pylint: disable=import-error
    OUTLIER_CHANNEL_INDICES,
    OUTLIER_STATION_INDICES,
    SPEED_OF_LIGHT,
)
from utils.data_sim import (  # pylint: disable=import-error
    generate_calibrator_data,
    init_config,
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
        freq = f["sol000/amplitude000/freq"][:]
        amp = f["sol000/amplitude000/val"][0]  # (ant, freq, pol)
        phase = f["sol000/phase000/val"][0]  # (ant, freq, pol)

    return pols, freq, amp, phase


def reference_phase_to_refant(phase, refant):
    """Remove the phase gauge freedom by zeroing the refant phase"""
    return phase - phase[[refant], :]


def normalised_rmse(actual, expected):
    """RMSE of (actual - expected), as a fraction of the band mean"""
    return np.sqrt(np.mean((actual - expected) ** 2)) / np.mean(expected)


def wrapped_phase_rmse(actual, expected):
    """RMSE of phase differences across one station's channels, wrapped
    to (-pi, pi]"""
    phase_error = np.angle(np.exp(1j * (actual - expected)))
    return np.sqrt(np.mean(phase_error**2))


def injected_cable_delays():
    """Per-station delays (seconds) injected via the cable length errors"""
    return np.loadtxt(CABLE_DELAYS, dtype=np.float64) / SPEED_OF_LIGHT


def injected_outlier_mask(n_stations, n_channels):
    """Boolean (station, channel) mask of the injected gain outliers"""
    mask = np.zeros((n_stations, n_channels), dtype=bool)
    mask[np.ix_(OUTLIER_STATION_INDICES, OUTLIER_CHANNEL_INDICES)] = True
    return mask


def validate_inst_gaintable(output_dir, temp_path, field_id, refant=0):
    amp_rmse_threshold = 0.06
    phase_rmse_threshold = np.deg2rad(6)

    expected_pols, expected_freq, expected_amp, expected_phase = (
        read_h5parm_gains(temp_path / "sim_gaintable.h5parm")
    )
    actual_pols, actual_freq, actual_amp, actual_phase = read_h5parm_gains(
        output_dir / f"{field_id}_inst.gaintable.h5parm"
    )

    np.testing.assert_allclose(
        actual_freq,
        expected_freq,
        err_msg="Frequencies don't match between INST and simulated gaintable",
    )

    delay_phase_ramp = (
        2
        * np.pi
        * expected_freq[np.newaxis, :]
        * injected_cable_delays()[:, np.newaxis]
    )

    n_stations, n_channels, _ = actual_amp.shape
    expected_flagged = injected_outlier_mask(n_stations, n_channels)

    for pol_name in ("XX", "YY"):
        expected_pol_idx = expected_pols.index(pol_name)
        actual_pol_idx = actual_pols.index(pol_name)

        expected_amp_pol = expected_amp[:, :, expected_pol_idx]
        expected_phase_pol = reference_phase_to_refant(
            expected_phase[:, :, expected_pol_idx] + delay_phase_ramp, refant
        )

        actual_amp_pol = actual_amp[:, :, actual_pol_idx]
        actual_phase_pol = reference_phase_to_refant(
            actual_phase[:, :, actual_pol_idx], refant
        )

        np.testing.assert_array_equal(
            np.isnan(actual_amp_pol),
            expected_flagged,
            err_msg=f"Flagged gains for {pol_name} do not match",
        )

        for station in VALIDATION_STATIONS:
            amp_rmse = normalised_rmse(
                actual_amp_pol[station], expected_amp_pol[station]
            )
            phase_rmse = wrapped_phase_rmse(
                actual_phase_pol[station], expected_phase_pol[station]
            )

            logger.info(
                "%s station %02d bandpass RMSE: "
                "amplitude (normalised)=%.3e phase=%.2f deg",
                pol_name,
                station,
                amp_rmse,
                np.rad2deg(phase_rmse),
            )

            assert amp_rmse < amp_rmse_threshold, (
                f"{pol_name} station {station} normalised amplitude RMSE "
                f"{amp_rmse:.3e} exceeds threshold {amp_rmse_threshold:.1e}"
            )
            assert phase_rmse < phase_rmse_threshold, (
                f"{pol_name} station {station} phase RMSE "
                f"{np.rad2deg(phase_rmse):.2f} deg exceeds threshold "
                f"{np.rad2deg(phase_rmse_threshold):.2f} deg"
            )


def validate_delay_stage(output_dir, ms_path, refant=0):
    delay_rmse_threshold = 5e-10

    expected_delays = injected_cable_delays()
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

    clean_stations = np.setdiff1d(
        np.arange(len(expected_delays)), OUTLIER_STATION_INDICES
    )

    for pol_idx, pol_name in enumerate(("XX", "YY")):
        delay_error = (
            actual_delay[clean_stations, pol_idx]
            - expected_delays[clean_stations]
        )
        delay_rmse = np.sqrt(np.mean(delay_error**2))
        logger.info("%s delay RMSE: %.3e s", pol_name, delay_rmse)

        assert delay_rmse < delay_rmse_threshold, (
            f"{pol_name} delay RMSE {delay_rmse:.3e}s over uncorrupted "
            f"stations exceeds threshold {delay_rmse_threshold:.1e}s"
        )


class INSTClibration(unittest.TestCase):
    def test_instrumental_calibration_integration(self):
        """Run integration test for Instrumental calibration Pipeline"""
        logger.info(
            "Run integration test for Instrumental calibration Pipeline"
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

            lsm_path = migrate_sky_model(SKY_MODEL, temp_path)
            inst_config_path = init_config(
                INST_CAL_CONFIG,
                temp_path,
                ms_path=input_ms_path,
                lsm_path=lsm_path,
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
