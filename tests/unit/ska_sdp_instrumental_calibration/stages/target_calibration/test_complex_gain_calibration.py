import pytest
from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.target_calibration.complex_gain_calibration import (  # noqa: E501
    PlotConfig,
    TargetRunSolverConfig,
    complex_gain_calibration_stage,
)
from ska_sdp_instrumental_calibration.xarray_processors import with_chunks


@pytest.fixture
def upstream_output():
    uo = UpstreamOutput()
    uo["gaintable"] = Mock(name="gaintable")
    uo["vis"] = Mock(name="vis")
    uo["corrected_vis"] = Mock(name="corrected_vis")
    uo["modelvis"] = Mock(name="modelvis")
    uo["chunks"] = Mock(name="chunks")
    uo["timeslice"] = 1
    return uo


@pytest.fixture
def run_solver_config():
    return TargetRunSolverConfig(
        refant=2,
        niter=1,
        tol=1e-6,
        crosspol=False,
    )


@pytest.fixture
def plot_config():
    return PlotConfig(
        plot_table=False,
        fixed_axis=False,
    )


@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.parse_antenna",
    return_value=3,
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.run_solver",
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.Solver",
)
@pytest.mark.parametrize("visibility_key_attr", ["vis", "corrected_vis"])
def test_should_perform_complex_gain_calibration(
    solver_factory_mock,
    run_solver_mock,
    parse_ref_ant_mock,
    upstream_output,
    run_solver_config,
    plot_config,
    visibility_key_attr,
):
    initial_gaintable = upstream_output.gaintable
    initial_gaintable.pipe.return_value = initial_gaintable
    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock
    solver_factory_mock.get_solver.return_value = "SOLVER"

    out = complex_gain_calibration_stage(
        upstream_output,
        run_solver_config=run_solver_config,
        plot_config=plot_config,
        visibility_key=visibility_key_attr,
        export_gaintable=False,
        _qa_dir_="/out",
    )

    parse_ref_ant_mock.assert_called_once_with(
        2,
        initial_gaintable.configuration.names,
    )
    initial_gaintable.pipe.assert_called_once_with(
        with_chunks, upstream_output["chunks"]
    )
    solver_factory_mock.get_solver.assert_called_once_with(
        solver="gain_substitution",
        niter=1,
        refant=3,
        phase_only=True,
        tol=1e-06,
        crosspol=False,
        timeslice=1,
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output[visibility_key_attr],
        modelvis=upstream_output.modelvis,
        gaintable=initial_gaintable,
        solver="SOLVER",
    )
    assert out.gaintable == gaintable_mock
    assert out["calibration_purpose"] == "gains"


@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.get_gaintables_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.h5exp.export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.PlotGaintableTime"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.parse_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration"
    ".complex_gain_calibration.Solver",
)
def test_should_export_gaintable_with_proper_suffix(
    solver_factory_mock,
    run_solver_mock,
    parse_reference_antenna_mock,
    plot_gaintable_time_mock,
    export_gaintable_mock,
    get_gaintables_path_mock,
    get_plot_path_mock,
    delayed_mock,
    upstream_output,
    run_solver_config,
    plot_config,
):
    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock
    plot_gaintable_time_mock.return_value = plot_gaintable_time_mock
    parse_reference_antenna_mock.return_value = 2
    plot_config.plot_table = True
    plot_config.fixed_axis = True

    actual_output = complex_gain_calibration_stage(
        upstream_output,
        _qa_dir_="/output/path",
        run_solver_config=run_solver_config,
        visibility_key="corrected_vis",
        plot_config=plot_config,
        export_gaintable=True,
    )

    get_gaintables_path_mock.assert_called_once_with(
        "/output/path", "complex_gain.gaintable.h5parm"
    )

    export_gaintable_mock.assert_called_once_with(
        gaintable_mock,
        get_gaintables_path_mock.return_value,
    )

    get_plot_path_mock.assert_called_once_with("/output/path", "complex_gain")
    plot_gaintable_time_mock.assert_called_once_with(
        path_prefix=get_plot_path_mock.return_value, refant=2
    )
    plot_gaintable_time_mock.plot.assert_called_once_with(
        gaintable_mock,
        figure_title="Complex Gain",
        fixed_axis=True,
    )

    delayed_mock.assert_called_once_with(export_gaintable_mock)

    assert actual_output.gaintable == gaintable_mock
