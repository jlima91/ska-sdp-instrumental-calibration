from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    export_visibilities_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.export_visibilities"
    ".apply_gaintable_to_dataset"
)
def test_should_apply_gaintable_on_vis(apply_gaintable_to_dataset_mock):
    upstream_output = UpstreamOutput()

    vis_mock = Mock(name="visibilities")
    modelvis_mock = Mock(name="model visibilities")
    gaintable_mock = Mock(name="gaintable")
    corrected_vis_mock = Mock(name="corrected vis")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = modelvis_mock
    upstream_output["gaintable"] = gaintable_mock
    apply_gaintable_to_dataset_mock.return_value = corrected_vis_mock

    result = export_visibilities_stage.stage_definition(upstream_output, "vis")

    apply_gaintable_to_dataset_mock.assert_called_once_with(
        vis_mock, gaintable_mock
    )
    assert result["corrected_vis"] == corrected_vis_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.export_visibilities"
    ".apply_gaintable_to_dataset"
)
def test_should_apply_gaintable_on_modelvis(apply_gaintable_to_dataset_mock):
    upstream_output = UpstreamOutput()

    vis_mock = Mock(name="visibilities")
    modelvis_mock = Mock(name="model visibilities")
    gaintable_mock = Mock(name="gaintable")
    corrected_modelvis_mock = Mock(name="corrected model vis")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = modelvis_mock
    upstream_output["gaintable"] = gaintable_mock
    apply_gaintable_to_dataset_mock.return_value = corrected_modelvis_mock

    result = export_visibilities_stage.stage_definition(
        upstream_output, "modelvis"
    )

    apply_gaintable_to_dataset_mock.assert_called_once_with(
        modelvis_mock, gaintable_mock
    )
    assert result["corrected_modelvis"] == corrected_modelvis_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.export_visibilities"
    ".apply_gaintable_to_dataset"
)
def test_should_apply_gaintable_on_vis_and_model_vis(
    apply_gaintable_to_dataset_mock,
):
    upstream_output = UpstreamOutput()

    vis_mock = Mock(name="visibilities")
    modelvis_mock = Mock(name="model visibilities")
    gaintable_mock = Mock(name="gaintable")
    corrected_vis_mock = Mock(name="corrected vis")
    corrected_modelvis_mock = Mock(name="corrected model vis")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = modelvis_mock
    upstream_output["gaintable"] = gaintable_mock
    apply_gaintable_to_dataset_mock.side_effect = [
        corrected_vis_mock,
        corrected_modelvis_mock,
    ]

    result = export_visibilities_stage.stage_definition(upstream_output, "all")
    apply_gaintable_to_dataset_mock.assert_has_calls(
        [
            call(vis_mock, gaintable_mock),
            call(modelvis_mock, gaintable_mock),
        ]
    )
    assert result["corrected_vis"] == corrected_vis_mock
    assert result["corrected_modelvis"] == corrected_modelvis_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.export_visibilities"
    ".apply_gaintable_to_dataset"
)
def test_should_not_apply_gaintable_on(apply_gaintable_to_dataset_mock):
    upstream_output = UpstreamOutput()

    vis_mock = Mock(name="visibilities")
    modelvis_mock = Mock(name="model visibilities")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = modelvis_mock
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage.stage_definition(upstream_output, None)

    apply_gaintable_to_dataset_mock.assert_not_called()
