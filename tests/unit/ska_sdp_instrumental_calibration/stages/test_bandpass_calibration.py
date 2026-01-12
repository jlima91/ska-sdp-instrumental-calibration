from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import bandpass_calibration_stage


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "bandpass_calibration": {
            "run_solver_config": {
                "solver": "jones_substitution",
                "refant": 0,
                "niter": 50,
                "phase_only": False,
                "tol": 1.0e-03,
                "crosspol": False,
                "normalise_gains": None,
            },
            "visibility_filters": {"uvdist": None, "exclude_baselines": None},
            "plot_config": {"plot_table": True, "fixed_axis": False},
            "visibility_key": "vis",
            "export_gaintable": True,
        }
    }

    assert bandpass_calibration_stage.config == expected_config


def test_bandpass_calibration_stage_is_required():
    assert bandpass_calibration_stage.is_required


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".parse_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".SolverFactory"
)
def test_should_perform_bandpass_calibration(
    solver_factory_mock, run_solver_mock, parse_ref_ant_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = Mock(name="initial_gaintable")
    upstream_output["gaintable"] = initable
    parse_ref_ant_mock.return_value = 3
    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
    }
    plot_config = {"plot_table": False, "fixed_axis": False}
    solver_factory_mock.get_solver.return_value = "SOLVER"

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=None,
        plot_config=plot_config,
        visibility_key="corrected_vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    parse_ref_ant_mock.assert_called_once_with(2, initable.configuration.names)

    solver_factory_mock.get_solver.assert_called_once_with(
        solver="solver",
        niter=1,
        refant=parse_ref_ant_mock.return_value,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output.corrected_vis,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="SOLVER",
    )

    assert actual_output.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".parse_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".PlotGaintableFrequency"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".SolverFactory"
)
def test_should_plot_bp_gaintable_with_proper_suffix(
    solver_factory_mock,
    run_solver_mock,
    plot_gaintable_freq_mock,
    get_plots_path_mock,
    parse_ref_ant_mock,
):
    get_plots_path_mock.side_effect = [
        "/output/path/plots/bandpass",
        "/output/path/plots/bandpass_1",
    ]
    plot_gaintable_freq_mock.return_value = plot_gaintable_freq_mock
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")

    solver_factory_mock.get_solver.return_value = "SOLVER"
    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
    }
    plot_config = {"plot_table": True, "fixed_axis": True}
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=None,
        plot_config=plot_config,
        visibility_key="corrected_vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=None,
        plot_config=plot_config,
        visibility_key="corrected_vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    get_plots_path_mock.assert_has_calls(
        [call("/output/path", "bandpass"), call("/output/path", "bandpass_1")]
    )
    plot_gaintable_freq_mock.assert_has_calls(
        [
            call(
                path_prefix="/output/path/plots/bandpass",
            ),
            call.plot(
                gaintable_mock,
                figure_title="Bandpass",
                fixed_axis=True,
                plot_all_stations=True,
            ),
            call(
                path_prefix="/output/path/plots/bandpass_1",
            ),
            call.plot(
                gaintable_mock,
                figure_title="Bandpass",
                fixed_axis=True,
                plot_all_stations=True,
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".parse_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".get_gaintables_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".SolverFactory"
)
def test_should_export_gaintable_with_proper_suffix(
    solver_factory_mock,
    run_solver_mock,
    export_gaintable_mock,
    get_gaintables_path_mock,
    delayed_mock,
    parse_ref_ant_mock,
):
    get_gaintables_path_mock.side_effect = [
        "/output/path/gaintables/bandpass.gaintable.h5parm",
        "/output/path/gaintables/bandpass_1.gaintable.h5parm",
    ]
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    solver_factory_mock.get_solver.return_value = "SOLVER"

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
    }
    plot_config = {"plot_table": False, "fixed_axis": True}
    gaintable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = gaintable_mock
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=None,
        plot_config=plot_config,
        visibility_key="corrected_vis",
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=None,
        plot_config=plot_config,
        visibility_key="corrected_vis",
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    get_gaintables_path_mock.assert_has_calls(
        [
            call("/output/path", "bandpass.gaintable.h5parm"),
            call("/output/path", "bandpass_1.gaintable.h5parm"),
        ]
    )

    export_gaintable_mock.assert_has_calls(
        [
            call(
                gaintable_mock,
                "/output/path/gaintables/bandpass.gaintable.h5parm",
            ),
            call(
                gaintable_mock,
                "/output/path/gaintables/bandpass_1.gaintable.h5parm",
            ),
        ]
    )

    delayed_mock.assert_has_calls(
        [call(export_gaintable_mock), call(export_gaintable_mock)]
    )


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".parse_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".SolverFactory"
)
def test_should_not_use_corrected_vis_when_config_is_false(
    solver_factory_mock, run_solver_mock, parse_ref_ant_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = Mock(name="initial_gaintable")
    upstream_output["gaintable"] = initable
    parse_ref_ant_mock.return_value = 3
    solver_factory_mock.get_solver.return_value = "SOLVER"
    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
    }
    plot_config = {"plot_table": False, "fixed_axis": False}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual_output = bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=None,
        plot_config=plot_config,
        visibility_key="vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    solver_factory_mock.get_solver.assert_called_once_with(
        solver="solver",
        niter=1,
        refant=parse_ref_ant_mock.return_value,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="SOLVER",
    )

    assert actual_output.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".VisibilityFilter"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".parse_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".SolverFactory"
)
def test_should_apply_uvrange_and_bandpass_filters_before_run_solver(
    solver_factory_mock,
    run_solver_mock,
    parse_ref_ant_mock,
    visibility_filter_mock,
):
    upstream_output = UpstreamOutput()
    mock_vis = Mock(name="vis")
    mock_vis.assign.return_value = mock_vis

    upstream_output["vis"] = mock_vis
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initable = Mock(name="initial_gaintable")
    upstream_output["gaintable"] = initable
    parse_ref_ant_mock.return_value = 3
    solver_factory_mock.get_solver.return_value = "SOLVER"
    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
    }
    plot_config = {"plot_table": False, "fixed_axis": False}

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    bandpass_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_filters=dict(uvdist=">500m", baselines="ANT1&ANT2"),
        plot_config=plot_config,
        visibility_key="vis",
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    visibility_filter_mock.filter.assert_called_once_with(
        dict(uvdist=">500m", baselines="ANT1&ANT2"), upstream_output.vis
    )

    run_solver_mock.assert_called_once_with(
        vis=visibility_filter_mock.filter.return_value,
        modelvis=upstream_output.modelvis,
        gaintable=initable,
        solver="SOLVER",
    )
