import os
import sys
import tempfile
from pathlib import Path

import pytest

from ska_sdp_instrumental_calibration.instrumental_calibration import (
    ska_sdp_instrumental_calibration,
)

from . import resources

#  IMPORTANT: Please don't change the order of E2E, as it will
#  change the default singleton stage configuration.


@pytest.fixture
def clean_argv():
    original_argv = sys.argv.copy()
    yield
    sys.argv = original_argv


@pytest.mark.order(-1)
def test_should_run_inst_and_generate_required_files(clean_argv):
    with tempfile.TemporaryDirectory() as temp_dir:
        print(temp_dir)
        test_resources = resources.init_data(temp_dir)
        sys.argv = [
            "ska-sdp-instrumental-calibration",
            "run",
            "--no-unique-output-subdir",
            "--config",
            test_resources.config,
            "--input",
            test_resources.ms_file,
            "--output",
            f"{temp_dir}/output",
            "--set",
            "parameters.predict_vis.gleamfile",
            test_resources.gleamdata,
            "--set",
            "parameters.predict_vis.eb_ms",
            test_resources.eb_ms,
        ]

        ska_sdp_instrumental_calibration()

        assert os.path.exists(f"{temp_dir}/output/demo.ms_fid0_ddid0/")
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/bandpass.gaintable.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/"
            "bandpass_initialisation.gaintable.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/channel_rm.gaintable.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/delay.clock.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/delay.gaintable.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/gain_flag.gaintable.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/gaintables/ionospheric_delay.gaintable.h5parm"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/bandpass-all_station_amp_vs_freq.png"
        )
        assert os.path.exists(f"{temp_dir}/output/plots/bandpass-amp-freq.png")
        assert os.path.exists(
            f"{temp_dir}/output/plots/bandpass-phase-freq.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/channel_rm-amp-freq.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/channel_rm-phase-freq.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/channel_rm-rm-station-LOWBD2_344.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/"
            "curve_fit_gain-curve-amp-phase_freq-LOWBD2_344-LOWBD2_347.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/"
            "curve_fit_gain-curve-amp-phase_freq-LOWBD2_348-LOWBD2_351.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/"
            "curve_fit_gain-curve-amp-phase_freq-LOWBD2_352-LOWBD2_429.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/"
            "curve_fit_gain-curve-amp-phase_freq-LOWBD2_430-LOWBD2_433.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/"
            "curve_fit_gain-curve-amp-phase_freq-LOWBD2_464-LOWBD2_467.png"
        )
        assert os.path.exists(f"{temp_dir}/output/plots/delay-amp-freq.png")
        assert os.path.exists(f"{temp_dir}/output/plots/delay-phase-freq.png")
        assert os.path.exists(
            f"{temp_dir}/output/plots/delay_station_delay.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/gain_flagging"
            "-weights_freq-LOWBD2_344-LOWBD2_433.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/gain_flagging"
            "-weights_freq-LOWBD2_464-LOWBD2_467.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/plots/ionospheric_delay-phase-freq.png"
        )
        assert os.path.exists(
            f"{temp_dir}/output/visibilities/corrected_vis.ms/"
        )
        assert os.path.exists(f"{temp_dir}/output/visibilities/modelvis.ms/")
        assert os.path.exists(f"{temp_dir}/output/inst.gaintable.h5parm")
        assert any(
            Path(f"{temp_dir}/output/").glob(
                "ska_sdp_instrumental_calibration*.cli.yml"
            )
        )
        assert any(
            Path(f"{temp_dir}/output/").glob(
                "ska_sdp_instrumental_calibration*.config.yml"
            )
        )
        assert any(
            Path(f"{temp_dir}/output/").glob(
                "ska_sdp_instrumental_calibration*.log"
            )
        )
        assert os.path.exists(f"{temp_dir}/output/sky_model.csv")
