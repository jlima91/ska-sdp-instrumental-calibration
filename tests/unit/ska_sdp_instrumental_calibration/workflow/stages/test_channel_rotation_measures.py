from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow import generate_channel_rm_stage


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.load_data_stage"
)
def test_should_generate_channel_rm_using_load_data_fchunk(
    load_data_mock, model_rotation_mock, run_solver_mock
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initialtable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = initialtable_mock
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}

    load_data_mock.config = {"load_data": {"fchunk": 30}}
    model_rotation_gaintable_mock = Mock(name="model rotation gaintable")
    model_rotation_gaintable_mock.chunk.return_value = (
        model_rotation_gaintable_mock
    )
    model_rotation_mock.return_value = model_rotation_gaintable_mock
    run_solver_gaintable_mock = Mock("run solver gaintable")
    run_solver_mock.return_value = run_solver_gaintable_mock

    output = generate_channel_rm_stage.stage_definition(
        upstream_output, fchunk=-1, run_solver_config=run_solver_config
    )

    model_rotation_mock.assert_called_once_with(
        initialtable_mock, plot_sample=True
    )

    model_rotation_gaintable_mock.chunk.assert_called_once_with(
        {"frequency": 30}
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=model_rotation_gaintable_mock,
        solver="solver",
        niter=1,
        refant=2,
    )

    assert output.gaintable == run_solver_gaintable_mock


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
    model_rotation_mock, run_solver_mock
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initialtable_mock = Mock(name="gaintable")
    upstream_output["gaintable"] = initialtable_mock
    run_solver_config = {"solver": "solver", "niter": 1, "refant": 2}

    model_rotation_gaintable_mock = Mock(name="model rotation gaintable")
    model_rotation_gaintable_mock.chunk.return_value = (
        model_rotation_gaintable_mock
    )
    model_rotation_mock.return_value = model_rotation_gaintable_mock
    run_solver_gaintable_mock = Mock("run solver gaintable")
    run_solver_mock.return_value = run_solver_gaintable_mock

    output = generate_channel_rm_stage.stage_definition(
        upstream_output, fchunk=32, run_solver_config=run_solver_config
    )

    model_rotation_mock.assert_called_once_with(
        initialtable_mock, plot_sample=True
    )

    model_rotation_gaintable_mock.chunk.assert_called_once_with(
        {"frequency": 32}
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=model_rotation_gaintable_mock,
        solver="solver",
        niter=1,
        refant=2,
    )

    assert output.gaintable == run_solver_gaintable_mock
