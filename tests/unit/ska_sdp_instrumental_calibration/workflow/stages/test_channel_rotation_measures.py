from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    generate_channel_rm_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.dask"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.model_rotations"
)
def test_should_generate_channel_rm_using_load_data_fchunk(
    model_rotations_mock,
    run_solver_mock,
    dask_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    model_rotated_gaintable = Mock(name="model rotated gaintable")
    model_rotations_mock.return_value = model_rotated_gaintable

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    result = generate_channel_rm_stage.stage_definition(
        upstream_output, fchunk=-1, run_solver_config=run_solver_config
    )

    model_rotations_mock.assert_called_once_with(
        initial_table_mock, plot_sample=True
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output["vis"],
        modelvis=upstream_output["modelvis"],
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
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".dask"
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
    model_rotations_mock, run_solver_mock, dask_mock
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    chunked_table_mock = Mock(name="chunked gaintable")
    initial_table_mock.chunk.return_value = chunked_table_mock

    model_rotated_gaintable = Mock(name="model rotated gaintable")
    model_rotations_mock.return_value = model_rotated_gaintable

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}
    result = generate_channel_rm_stage.stage_definition(
        upstream_output, fchunk=40, run_solver_config=run_solver_config
    )

    initial_table_mock.chunk.assert_called_once_with({"frequency": 40})
    model_rotations_mock.assert_called_once_with(
        chunked_table_mock, plot_sample=True
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
