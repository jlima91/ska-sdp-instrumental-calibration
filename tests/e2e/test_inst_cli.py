import sys
import tempfile
from pathlib import Path

import pytest
from mock import patch

from ska_sdp_instrumental_calibration.instrumental_calibration import (
    ska_sdp_instrumental_calibration,
)

from . import resources

#  IMPORTANT: Please don't change the order of E2E, as it will
#  change the default singleton stage configuration.


@pytest.mark.order(-1)
def test_should_run_inst_and_generate_required_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(temp_dir)
        test_resources = resources.init_data(temp_dir)
        testargs = [
            "ska-sdp-instrumental-calibration",
            "run",
            "--no-unique-output-subdir",
            "--config",
            test_resources.config,
            "--output",
            f"{temp_dir}/output",
            "--set",
            "parameters.predict_vis.lsm_csv_path",
            test_resources.lsm_csv,
            *test_resources.ms_files,
        ]

        with patch.object(sys, "argv", testargs):
            ska_sdp_instrumental_calibration()

        output_dir = Path(f"{temp_dir}/output")
        qa_dir = output_dir

        assert (output_dir / ".cache" / "a_test.ms_fid0_ddid0").exists()
        assert (output_dir / ".cache" / "test.ms_fid0_ddid0").exists()

        assert (output_dir / "visibilities/test/corrected.ms/").exists()
        assert (output_dir / "visibilities/a_test/corrected.ms/").exists()
        assert (output_dir / "visibilities/test/modelvis.ms/").exists()
        assert (output_dir / "visibilities/a_test/modelvis.ms/").exists()

        assert (output_dir / "CAL_FIELD_gaintable.h5parm").exists()

        assert any(qa_dir.glob("ska_sdp_instrumental_calibration*.cli.yaml"))
        assert any(
            qa_dir.glob("ska_sdp_instrumental_calibration*.config.yaml")
        )
        assert any(qa_dir.glob("ska_sdp_instrumental_calibration*.log"))
        assert (qa_dir / "sky/test/sky_model.csv").exists()
        assert (qa_dir / "sky/a_test/sky_model.csv").exists()

        test_qa_plots = {
            qa_file.name for qa_file in (qa_dir / "plots/test").glob("*.png")
        }
        a_test_qa_plots = {
            qa_file.name for qa_file in (qa_dir / "plots/a_test").glob("*.png")
        }

        test_qa_gaintables = {
            qa_file.name
            for qa_file in (qa_dir / "gaintables/test").glob("*.h5parm")
        }
        a_test_qa_gaintables = {
            qa_file.name
            for qa_file in (qa_dir / "gaintables/a_test").glob("*.h5parm")
        }

        assert len(test_qa_plots) == len(a_test_qa_plots)
        assert len(test_qa_plots) == 21
        assert len(test_qa_gaintables) == len(a_test_qa_gaintables)
        assert len(test_qa_gaintables) == 6

        for stage in [
            "channel_rm",
            "bandpass_initialisation",
            "gain_flag",
            "delay",
            "bandpass",
        ]:
            gaintable_file = f"{stage}.gaintable.h5parm"
            assert gaintable_file in test_qa_gaintables

        assert "delay.clock.h5parm" in test_qa_gaintables
