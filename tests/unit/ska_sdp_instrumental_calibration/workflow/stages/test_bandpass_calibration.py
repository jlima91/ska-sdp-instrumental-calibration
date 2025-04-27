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
    plot_config = {"plot_table": False, "fixed_axis": False}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        _output_dir_="/output/path",
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
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_plot_bp_gaintable(
    run_solver_mock, plot_gaintable_mock, dask_delayed_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")

    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    plot_config = {"plot_table": True, "fixed_axis": True}
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        _output_dir_="/output/path",
    )

    plot_gaintable_mock.assert_called_once_with(
        gaintable_mock,
        "/output/path/bandpass",
        figure_title="Bandpass",
        fixed_axis=True,
        all_station_plot=True,
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_call_delayed_run_solver_if_gain_table_is_a_delayed_object(
    run_solver_mock, plot_gaintable_mock, dask_delayed_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = Mock(name="initial_gaintable")
    initable.dask = lambda x: x

    upstream_output["gaintable"] = initable
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    plot_config = {"plot_table": False, "fixed_axis": False}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        _output_dir_="/output/path",
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
