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
            "parameters.predict_vis.gleamfile",
            test_resources.gleamdata,
            "--set",
            "parameters.predict_vis.eb_ms",
            test_resources.eb_ms,
            *test_resources.ms_files,
        ]

        with patch.object(sys, "argv", testargs):
            ska_sdp_instrumental_calibration()

        output_dir = Path(f"{temp_dir}/output")
        qa_dir = output_dir / "sdm/logs/01-inst"

        assert (output_dir / "a_demo.ms_fid0_ddid0").exists()
        assert (output_dir / "demo.ms_fid0_ddid0").exists()

        assert (output_dir / "visibilities/demo/corrected.ms/").exists()
        assert (output_dir / "visibilities/a_demo/corrected.ms/").exists()
        assert (output_dir / "visibilities/demo/modelvis.ms/").exists()
        assert (output_dir / "visibilities/a_demo/modelvis.ms/").exists()

        # [TODO] Update "unknown" with the correct field name
        # once test data is fixed
        assert (output_dir / "unknown_gaintable.h5parm").exists()

        assert any(qa_dir.glob("ska_sdp_instrumental_calibration*.cli.yaml"))
        assert any(
            qa_dir.glob("ska_sdp_instrumental_calibration*.config.yaml")
        )
        assert any(qa_dir.glob("ska_sdp_instrumental_calibration*.log"))
        assert (qa_dir / "sky/demo/sky_model.csv").exists()
        assert (qa_dir / "sky/a_demo/sky_model.csv").exists()

        demo_qa_plots = {
            qa_file.name for qa_file in (qa_dir / "plots/demo").glob("*.png")
        }
        a_demo_qa_plots = {
            qa_file.name for qa_file in (qa_dir / "plots/a_demo").glob("*.png")
        }

        demo_qa_gaintables = {
            qa_file.name
            for qa_file in (qa_dir / "gaintables/demo").glob("*.h5parm")
        }
        a_demo_qa_gaintables = {
            qa_file.name
            for qa_file in (qa_dir / "gaintables/a_demo").glob("*.h5parm")
        }

        assert len(demo_qa_plots) == len(a_demo_qa_plots)
        assert len(demo_qa_gaintables) == len(a_demo_qa_gaintables)

        assert len(demo_qa_gaintables) == 7

        for stage in [
            "channel_rm",
            "bandpass_initialisation",
            "gain_flag",
            "delay",
            "ionospheric_delay",
            "bandpass",
        ]:
            gaintable_file = f"{stage}.gaintable.h5parm"
            assert gaintable_file in demo_qa_gaintables

        assert "delay.clock.h5parm" in demo_qa_gaintables
