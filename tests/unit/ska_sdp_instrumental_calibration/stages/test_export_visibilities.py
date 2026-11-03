import numpy as np
import xarray as xr
from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import export_visibilities_stage
from ska_sdp_instrumental_calibration.stages.export_visibilities import (
    _export_visibility_to_ms,
)


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "export_visibilities": {
            "data_to_export": "all",
            "apply_gaintable_to_vis": True,
        },
    }

    assert export_visibilities_stage.__stage__.config == expected_config


def test_export_visibilities_stage_is_optional():
    assert not export_visibilities_stage.__stage__.is_enabled


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.get_visibilities_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities._export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.export_visibilities.delayed",
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
    upstream_output["ms_prefix"] = "ms_prefix"
    vis_mock = Mock(name="original_vis")
    corrected_vis_mock = Mock(name="corrected_vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["gaintable"] = gaintable_mock
    apply_gaintable_to_dataset_mock.return_value = corrected_vis_mock

    result = export_visibilities_stage(upstream_output, "./", "vis", True)

    apply_gaintable_to_dataset_mock.assert_called_once_with(
        vis_mock, upstream_output["gaintable"], inverse=True
    )
    get_visibilities_path_mock.assert_called_once_with(
        "./", "ms_prefix/corrected.ms"
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
    "export_visibilities._export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.export_visibilities.delayed",
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
    get_visibilities_path_mock.return_value = "./visibilities/modelvis.ms"
    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    model_vis = Mock(name="model vis")
    vis_mock = Mock(name="vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = model_vis
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage(upstream_output, "./", "modelvis", False)
    get_visibilities_path_mock.assert_called_once_with(
        "./", "ms_prefix/modelvis.ms"
    )
    export_mock.assert_called_once_with(
        "./visibilities/modelvis.ms", [model_vis]
    )
    apply_gaintable_to_dataset_mock.assert_not_called()


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.get_visibilities_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities._export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.export_visibilities.delayed",
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
        "./visibilities/raw_vis.ms",
        "./visibilities/modelvis.ms",
    ]
    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    model_vis = Mock(name="model vis")
    vis_mock = Mock(name="vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = model_vis
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage(upstream_output, "./", "all", False)

    get_visibilities_path_mock.assert_has_calls(
        [
            call("./", "ms_prefix/raw.ms"),
            call("./", "ms_prefix/modelvis.ms"),
        ]
    )
    export_mock.assert_has_calls(
        [
            call("./visibilities/raw_vis.ms", [upstream_output["vis"]]),
            call(
                "./visibilities/modelvis.ms",
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
    "export_visibilities._export_visibility_to_ms"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.export_visibilities.delayed",
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
        "./visibilities/raw_vis.ms",
        "./visibilities/modelvis.ms",
        "./visibilities/raw_vis_1.ms",
        "./visibilities/modelvis_1.ms",
    ]
    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    model_vis = Mock(name="model vis")
    vis_mock = Mock(name="vis")
    gaintable_mock = Mock(name="gaintable")
    upstream_output["vis"] = vis_mock
    upstream_output["modelvis"] = model_vis
    upstream_output["gaintable"] = gaintable_mock

    export_visibilities_stage(
        upstream_output,
        "./",
        "all",
        False,
    )
    export_visibilities_stage(upstream_output, "./", "all", False)
    get_visibilities_path_mock.assert_has_calls(
        [
            call("./", "ms_prefix/raw.ms"),
            call("./", "ms_prefix/modelvis.ms"),
            call("./", "ms_prefix/raw_1.ms"),
            call("./", "ms_prefix/modelvis_1.ms"),
        ]
    )

    export_mock.assert_has_calls(
        [
            call("./visibilities/raw_vis.ms", [upstream_output["vis"]]),
            call(
                "./visibilities/modelvis.ms",
                [upstream_output["modelvis"]],
            ),
            call("./visibilities/raw_vis_1.ms", [upstream_output["vis"]]),
            call(
                "./visibilities/modelvis_1.ms",
                [upstream_output["modelvis"]],
            ),
        ]
    )
    apply_gaintable_to_dataset_mock.assert_not_called()


@patch(
    "ska_sdp_instrumental_calibration.stages."
    "export_visibilities.export_visibility_to_ms"
)
def test_should_add_time_centroid_before_exporting_to_ms(export_mock):
    time = np.array([10.0, 20.0])
    vis = xr.Dataset(
        {
            "vis": (
                ["time", "baselines", "frequency", "polarisation"],
                np.zeros((2, 3, 4, 2), dtype=complex),
            )
        },
        coords={
            "time": time,
            "polarisation": ["XX", "YY"],
        },
    )

    _export_visibility_to_ms("./visibilities/vis.ms", [vis])

    export_mock.assert_called_once()
    msname, (exported_vis,) = export_mock.call_args.args
    assert msname == "./visibilities/vis.ms"
    assert exported_vis.time_centroid.dims == (
        "time",
        "baselines",
        "frequency",
    )
    np.testing.assert_array_equal(
        exported_vis.time_centroid.values,
        np.broadcast_to(time[:, None, None], (2, 3, 4)),
    )
    assert "time_centroid" not in vis
