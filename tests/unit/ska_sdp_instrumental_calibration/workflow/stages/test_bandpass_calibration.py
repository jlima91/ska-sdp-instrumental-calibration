from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    bandpass_calibration_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_perform_bandpass_calibration(run_solver_mock):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        flagging=False,
        plot_table=False,
        _output_dir_="/output/path",
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        solver="solver",
        niter=1,
        refant=2,
    )

    assert actual_output.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_plot_bp_gaintable(run_solver_mock, plot_gaintable_mock):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        flagging=False,
        plot_table=True,
        _output_dir_="/output/path",
    )

    plot_gaintable_mock.assert_called_once_with(
        gaintable_mock, "/output/path/bandpass"
    )
