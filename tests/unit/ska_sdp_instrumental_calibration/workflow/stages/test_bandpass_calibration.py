from mock import Mock, call, patch

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
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = "initial_gaintable"
    upstream_output["gaintable"] = initable
    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_config = {"plot_table": False, "fixed_axis": False}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        visibility_key="corrected_vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.corrected_vis,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="solver",
        niter=1,
        refant=2,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        jones_type="T",
        timeslice=None,
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
def test_should_plot_bp_gaintable_with_proper_suffix(
    run_solver_mock, plot_gaintable_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_config = {"plot_table": True, "fixed_axis": True}
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        visibility_key="corrected_vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        visibility_key="corrected_vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    plot_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "/output/path/bandpass",
                figure_title="Bandpass",
                fixed_axis=True,
                all_station_plot=True,
            ),
            call(
                gaintable_mock,
                "/output/path/bandpass_1",
                figure_title="Bandpass",
                fixed_axis=True,
                all_station_plot=True,
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_export_gaintable_with_proper_suffix(
    run_solver_mock, export_gaintable_mock, delayed_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_config = {"plot_table": False, "fixed_axis": True}
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        visibility_key="corrected_vis",
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        visibility_key="corrected_vis",
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    export_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "/output/path/bandpass.gaintable.h5parm",
            ),
            call(
                gaintable_mock,
                "/output/path/bandpass_1.gaintable.h5parm",
            ),
        ]
    )

    delayed_mock.assert_has_calls(
        [call(export_gaintable_mock), call(export_gaintable_mock)]
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.bandpass_calibration"
    ".run_solver"
)
def test_should_not_use_corrected_vis_when_config_is_false(run_solver_mock):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = "initial_gaintable"
    upstream_output["gaintable"] = initable
    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_config = {"plot_table": False, "fixed_axis": False}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        flagging=False,
        visibility_key="vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="solver",
        niter=1,
        refant=2,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        jones_type="T",
        timeslice=None,
    )

    assert actual_output.gaintable == gaintable_mock
