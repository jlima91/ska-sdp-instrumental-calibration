import pytest
from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.plot_x_dim import XDim_Time
from ska_sdp_instrumental_calibration.workflow.stages import target_calibration
from ska_sdp_instrumental_calibration.workflow.utils import with_chunks

complex_gain_calibration_stage = (
    target_calibration.complex_gain_calibration_stage
)


@pytest.fixture
def upstream_output():
    uo = UpstreamOutput()
    uo["vis"] = Mock(name="vis")
    uo["corrected_vis"] = Mock(name="corrected_vis")
    uo["modelvis"] = Mock(name="modelvis")
    uo["chunks"] = Mock(name="chunks")
    return uo


@pytest.fixture
def run_solver_config():
    return {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": True,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "timeslice": 0.5,
    }


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.parse_reference_antenna",
    return_value=3,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.create_gaintable_from_visibility",
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.target_solver.run_solver",
)
@pytest.mark.parametrize("visibility_key_attr", ["vis", "corrected_vis"])
def test_should_perform_complex_gain_calibration(
    run_solver_mock,
    create_gaintable_mock,
    parse_ref_ant_mock,
    upstream_output,
    run_solver_config,
    visibility_key_attr,
):
    initial_gaintable = Mock(name="initial_gaintable")
    initial_gaintable.pipe.return_value = initial_gaintable
    create_gaintable_mock.return_value = initial_gaintable
    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock
    plot_config = {"plot_table": False, "fixed_axis": False}

    out = complex_gain_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_key=visibility_key_attr,
        plot_config=plot_config,
        export_gaintable=False,
        _output_dir_="/out",
    )

    parse_ref_ant_mock.assert_called_once_with(2, initial_gaintable)
    create_gaintable_mock.assert_called_once_with(
        upstream_output[visibility_key_attr], timeslice=0.5, jones_type="G"
    )
    initial_gaintable.pipe.assert_called_once_with(
        with_chunks, upstream_output["chunks"]
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output[visibility_key_attr],
        modelvis=upstream_output.modelvis,
        gaintable=initial_gaintable,
        solver="solver",
        niter=1,
        refant=3,
        phase_only=True,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        timeslice=0.5,
    )
    assert out.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.get_gaintables_path"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.h5exp.export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.parse_reference_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.create_gaintable_from_visibility",
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".complex_gain_calibration.target_solver.run_solver"
)
def test_should_export_gaintable_with_proper_suffix(
    run_solver_mock,
    create_gaintable_mock,
    parse_ref_ant_mock,
    plot_gaintable_mock,
    export_gaintable_mock,
    get_gaintables_path_mock,
    get_plot_path_mock,
    delayed_mock,
    upstream_output,
    run_solver_config,
):

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock
    plot_config = {"plot_table": True, "fixed_axis": True}

    actual_output = complex_gain_calibration_stage.stage_definition(
        upstream_output,
        run_solver_config=run_solver_config,
        visibility_key="corrected_vis",
        plot_config=plot_config,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    get_gaintables_path_mock.assert_called_once_with(
        "/output/path", "complex_gain.gaintable.h5parm"
    )

    export_gaintable_mock.assert_called_once_with(
        gaintable_mock,
        get_gaintables_path_mock.return_value,
    )

    get_plot_path_mock.assert_called_once_with("/output/path", "complex_gain")
    plot_gaintable_mock.assert_called_once_with(
        gaintable_mock,
        get_plot_path_mock.return_value,
        figure_title="Complex Gain",
        fixed_axis=True,
        x_dim=XDim_Time,
    )

    delayed_mock.assert_called_once_with(export_gaintable_mock)

    assert actual_output.gaintable == gaintable_mock
