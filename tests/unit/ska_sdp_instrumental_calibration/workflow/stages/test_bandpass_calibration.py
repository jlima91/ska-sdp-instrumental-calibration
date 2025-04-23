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
    initable = "initial_gaintable"
    upstream_output["gaintable"] = initable
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output, run_solver_config=run_solver_config, flagging=False
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="solver",
        niter=1,
        refant=2,
    )

    assert actual_output.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_call_delayed_run_solver_if_gain_table_is_a_delayed_object(
    run_solver_mock, dask_delayed_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = Mock(name="initial_gaintable")
    initable.dask = lambda x: x

    upstream_output["gaintable"] = initable
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output, run_solver_config=run_solver_config, flagging=False
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="solver",
        niter=1,
        refant=2,
    )

    dask_delayed_mock.assert_called_once_with(run_solver_mock)

    assert actual_output.gaintable == gaintable_mock
