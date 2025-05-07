from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    generate_channel_rm_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.model_rotations"
)
def test_should_generate_channel_rm_using_initial_gaintable(
    model_rotations_mock,
    run_solver_mock,
    dask_delayed_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    model_rotated_gaintable = Mock(name="model rotated gaintable")
    model_rotations_mock.return_value = model_rotated_gaintable

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    result = generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=-1,
        peak_threshold=0.5,
        plot_table=False,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    model_rotations_mock.assert_called_once_with(
        initial_table_mock,
        peak_threshold=0.5,
        plot_sample=False,
        plot_path_prefix="/output/path/channel_rm",
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output["vis"],
        modelvis=upstream_output["modelvis"],
        gaintable=model_rotated_gaintable,
        solver="solver",
        niter=1,
        refant=2,
    )
    dask_delayed_mock.assert_has_calls(
        [
            call(model_rotations_mock),
            call(run_solver_mock),
        ]
    )

    assert result["gaintable"] == solved_gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".model_rotations"
)
def test_should_generate_channel_rm_using_provided_fchunk(
    model_rotations_mock, run_solver_mock, delayed_mock
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    chunked_table_mock = Mock(name="chunked gaintable")
    initial_table_mock.chunk.return_value = chunked_table_mock

    model_rotated_gaintable = Mock(name="model rotated gaintable")
    model_rotations_mock.return_value = model_rotated_gaintable

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    result = generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        plot_table=False,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    initial_table_mock.chunk.assert_called_once_with({"frequency": 40})
    model_rotations_mock.assert_called_once_with(
        chunked_table_mock,
        peak_threshold=0.5,
        plot_sample=False,
        plot_path_prefix="/output/path/channel_rm",
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=model_rotated_gaintable,
        solver="solver",
        niter=1,
        refant=2,
    )
    delayed_mock.assert_has_calls(
        [
            call(model_rotations_mock),
            call(run_solver_mock),
        ]
    )

    assert result["gaintable"] == solved_gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".model_rotations"
)
def test_should_plot_channel_rm_gaintable_with_proper_suffix(
    model_rotations_mock, run_solver_mock, delayed_mock, plot_gaintable_mock
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    chunked_table_mock = Mock(name="chunked gaintable")
    initial_table_mock.chunk.return_value = chunked_table_mock

    model_rotated_gaintable = Mock(name="model rotated gaintable")
    model_rotations_mock.return_value = model_rotated_gaintable

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        plot_table=True,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    upstream_output["gaintable"] = initial_table_mock
    generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        plot_table=True,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    model_rotations_mock.assert_has_calls(
        [
            call(
                chunked_table_mock,
                peak_threshold=0.5,
                plot_sample=True,
                plot_path_prefix="/output/path/channel_rm",
            ),
            call(
                chunked_table_mock,
                peak_threshold=0.5,
                plot_sample=True,
                plot_path_prefix="/output/path/channel_rm_1",
            ),
        ]
    )

    plot_gaintable_mock.assert_has_calls(
        [
            call(
                solved_gaintable_mock,
                "/output/path/channel_rm",
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            ),
            call(
                solved_gaintable_mock,
                "/output/path/channel_rm_1",
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            ),
        ]
    )
