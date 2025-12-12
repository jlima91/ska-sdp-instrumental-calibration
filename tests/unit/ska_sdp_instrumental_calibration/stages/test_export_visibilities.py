from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import export_visibilities_stage


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "export_visibilities": {
            "data_to_export": "all",
            "apply_gaintable_to_vis": True,
        },
    }

    assert export_visibilities_stage.config == expected_config


def test_export_visibilities_stage_is_required():
    assert export_visibilities_stage.is_required


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.get_visibilities_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.apply_gaintable_to_dataset"
)
def test_should_export_corrected_vis_when_apply_gaintable_is_vis(
    apply_gaintable_to_dataset_mock,
    delayed_mock,
    export_mock,
    get_visibilities_path_mock,
):
    get_visibilities_path_mock.return_value = "./visibilities/corrected_vis.ms"
    upstream_output = UpstreamOutput()
    vis_mock = Mock(name="original_vis")
    corrected_vis_mock = Mock(name="corrected_vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["gaintable"] = gaintable_mock
    apply_gaintable_to_dataset_mock.return_value = corrected_vis_mock

    result = export_visibilities_stage.stage_definition(
        upstream_output, "vis", True, "./"
    )

    apply_gaintable_to_dataset_mock.assert_called_once_with(
        vis_mock, upstream_output["gaintable"], inverse=True
    )
    get_visibilities_path_mock.assert_called_once_with(
        "./", "corrected_vis.ms"
    )
    export_mock.assert_called_once_with(
        "./visibilities/corrected_vis.ms", [corrected_vis_mock]
    )
    assert result["corrected_vis"] == corrected_vis_mock


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.get_visibilities_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.apply_gaintable_to_dataset"
)
def test_should_export_model_vis(
    apply_gaintable_to_dataset_mock,
    delayed_mock,
    export_mock,
    get_visibilities_path_mock,
):
    get_visibilities_path_mock.return_value = (
        "./visibilities/corrected_modelvis.ms"
    )
    upstream_output = UpstreamOutput()
    model_vis = Mock(name="model vis")
    vis_mock = Mock(name="vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = model_vis
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage.stage_definition(
        upstream_output, "modelvis", False, "./"
    )
    get_visibilities_path_mock.assert_called_once_with(
        "./", "corrected_modelvis.ms"
    )
    export_mock.assert_called_once_with(
        "./visibilities/corrected_modelvis.ms", [model_vis]
    )
    apply_gaintable_to_dataset_mock.assert_not_called()


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.get_visibilities_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.apply_gaintable_to_dataset"
)
def test_should_export_both_vis_and_model_vis(
    apply_gaintable_to_dataset_mock,
    delayed_mock,
    export_mock,
    get_visibilities_path_mock,
):
    get_visibilities_path_mock.side_effect = [
        "./visibilities/corrected_vis.ms",
        "./visibilities/corrected_modelvis.ms",
    ]
    upstream_output = UpstreamOutput()
    model_vis = Mock(name="model vis")
    vis_mock = Mock(name="vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = model_vis
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage.stage_definition(
        upstream_output, "all", False, "./"
    )

    get_visibilities_path_mock.assert_has_calls(
        [
            call("./", "corrected_vis.ms"),
            call("./", "corrected_modelvis.ms"),
        ]
    )
    export_mock.assert_has_calls(
        [
            call("./visibilities/corrected_vis.ms", [upstream_output["vis"]]),
            call(
                "./visibilities/corrected_modelvis.ms",
                [upstream_output["modelvis"]],
            ),
        ]
    )
    apply_gaintable_to_dataset_mock.assert_not_called()


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.get_visibilities_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.apply_gaintable_to_dataset"
)
def test_should_maintain_call_count_and_add_suffix_for_exported_ms(
    apply_gaintable_to_dataset_mock,
    delayed_mock,
    export_mock,
    get_visibilities_path_mock,
):
    get_visibilities_path_mock.side_effect = [
        "./visibilities/corrected_vis.ms",
        "./visibilities/corrected_modelvis.ms",
        "./visibilities/corrected_vis_1.ms",
        "./visibilities/corrected_modelvis_1.ms",
    ]
    upstream_output = UpstreamOutput()
    model_vis = Mock(name="model vis")
    vis_mock = Mock(name="vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = model_vis
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage.stage_definition(
        upstream_output, "all", False, "./"
    )
    export_visibilities_stage.stage_definition(
        upstream_output, "all", False, "./"
    )

    get_visibilities_path_mock.assert_has_calls(
        [
            call("./", "corrected_vis.ms"),
            call("./", "corrected_modelvis.ms"),
            call("./", "corrected_vis_1.ms"),
            call("./", "corrected_modelvis_1.ms"),
        ]
    )

    export_mock.assert_has_calls(
        [
            call("./visibilities/corrected_vis.ms", [upstream_output["vis"]]),
            call(
                "./visibilities/corrected_modelvis.ms",
                [upstream_output["modelvis"]],
            ),
            call(
                "./visibilities/corrected_vis_1.ms", [upstream_output["vis"]]
            ),
            call(
                "./visibilities/corrected_modelvis_1.ms",
                [upstream_output["modelvis"]],
            ),
        ]
    )
    apply_gaintable_to_dataset_mock.assert_not_called()
