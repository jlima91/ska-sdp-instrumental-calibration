import numpy as np
import pytest
import xarray as xr
from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import export_visibilities_stage
from ska_sdp_instrumental_calibration.stages.export_visibilities import (
    _export_visibility_to_ms,
    _wait_for_ps_write,
)

MODULE = "ska_sdp_instrumental_calibration.stages.export_visibilities"


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "export_visibilities": {
            "data_to_export": "all",
            "apply_gaintable_to_vis": True,
            "export_format": "ps",
        },
    }

    assert export_visibilities_stage.__stage__.config == expected_config


def test_export_visibilities_stage_is_optional():
    assert not export_visibilities_stage.__stage__.is_enabled


@pytest.fixture
def upstream_output():
    upstream_output = UpstreamOutput()
    upstream_output["ms_prefix"] = "ms_prefix"
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["gaintable"] = Mock(name="gaintable")
    return upstream_output


@pytest.fixture
def mocks():
    """
    Patch the writers, apply_gaintable_to_dataset, and delayed so that
    deferred functions are called directly. get_visibilities_path returns
    "<output_dir>/visibilities/<file_prefix>".
    """
    with patch(f"{MODULE}.delayed", side_effect=lambda func: func), patch(
        f"{MODULE}.get_visibilities_path",
        side_effect=lambda output_dir, prefix: (
            f"{output_dir}/visibilities/{prefix}"
        ),
    ) as get_visibilities_path, patch(
        f"{MODULE}.apply_gaintable_to_dataset"
    ) as apply_gaintable_to_dataset, patch(
        f"{MODULE}.export_msv4_to_ps"
    ) as export_msv4_to_ps, patch(
        f"{MODULE}.msv4_to_visibility"
    ) as msv4_to_visibility, patch(
        f"{MODULE}._export_visibility_to_ms"
    ) as export_visibility_to_ms, patch(
        f"{MODULE}._wait_for_ps_write"
    ) as wait_for_ps_write:
        yield Mock(
            get_visibilities_path=get_visibilities_path,
            apply_gaintable_to_dataset=apply_gaintable_to_dataset,
            export_msv4_to_ps=export_msv4_to_ps,
            msv4_to_visibility=msv4_to_visibility,
            export_visibility_to_ms=export_visibility_to_ms,
            wait_for_ps_write=wait_for_ps_write,
        )


def test_should_export_corrected_vis_to_processing_set_by_default(
    upstream_output, mocks
):
    vis = upstream_output["vis"]
    corrected_vis = Mock(name="corrected_vis")
    mocks.apply_gaintable_to_dataset.return_value = corrected_vis

    result = export_visibilities_stage(upstream_output, ".", "vis", True)

    mocks.apply_gaintable_to_dataset.assert_called_once_with(
        vis, upstream_output["gaintable"], inverse=True
    )
    mocks.get_visibilities_path.assert_called_once_with(
        ".", "ms_prefix/corrected"
    )
    mocks.export_msv4_to_ps.assert_called_once_with(
        "./visibilities/ms_prefix/corrected.ps.zarr",
        corrected_vis,
        "ms_prefix_corrected",
    )
    # The write task is registered with the task manager
    mocks.wait_for_ps_write.assert_called_once_with(
        "./visibilities/ms_prefix/corrected.ps.zarr",
        mocks.export_msv4_to_ps.return_value,
    )
    mocks.export_visibility_to_ms.assert_not_called()
    assert result["corrected_vis"] == corrected_vis


def test_should_export_model_vis_to_msv2(upstream_output, mocks):
    modelvis = upstream_output["modelvis"]

    export_visibilities_stage(
        upstream_output, ".", "modelvis", False, export_format="ms"
    )

    mocks.get_visibilities_path.assert_called_once_with(
        ".", "ms_prefix/modelvis"
    )
    mocks.msv4_to_visibility.assert_called_once_with(modelvis)
    mocks.export_visibility_to_ms.assert_called_once_with(
        "./visibilities/ms_prefix/modelvis.ms",
        [mocks.msv4_to_visibility.return_value],
    )
    mocks.export_msv4_to_ps.assert_not_called()
    mocks.apply_gaintable_to_dataset.assert_not_called()


def test_should_export_vis_and_model_vis_in_both_formats(
    upstream_output, mocks
):
    vis, modelvis = upstream_output["vis"], upstream_output["modelvis"]

    export_visibilities_stage(
        upstream_output, ".", "all", False, export_format="both"
    )

    mocks.export_msv4_to_ps.assert_has_calls(
        [
            call("./visibilities/ms_prefix/raw.ps.zarr", vis, "ms_prefix_raw"),
            call(
                "./visibilities/ms_prefix/modelvis.ps.zarr",
                modelvis,
                "ms_prefix_modelvis",
            ),
        ]
    )
    mocks.msv4_to_visibility.assert_has_calls([call(vis), call(modelvis)])
    mocks.export_visibility_to_ms.assert_has_calls(
        [
            call(
                "./visibilities/ms_prefix/raw.ms",
                [mocks.msv4_to_visibility.return_value],
            ),
            call(
                "./visibilities/ms_prefix/modelvis.ms",
                [mocks.msv4_to_visibility.return_value],
            ),
        ]
    )
    mocks.apply_gaintable_to_dataset.assert_not_called()
    assert "corrected_vis" not in upstream_output


def test_should_maintain_call_count_and_add_suffix_for_exported_vis(
    upstream_output, mocks
):
    export_visibilities_stage(upstream_output, ".", "all", False)
    export_visibilities_stage(upstream_output, ".", "all", False)

    mocks.get_visibilities_path.assert_has_calls(
        [
            call(".", "ms_prefix/raw"),
            call(".", "ms_prefix/modelvis"),
            call(".", "ms_prefix/raw_1"),
            call(".", "ms_prefix/modelvis_1"),
        ]
    )
    mocks.export_msv4_to_ps.assert_has_calls(
        [
            call(
                "./visibilities/ms_prefix/raw.ps.zarr",
                upstream_output["vis"],
                "ms_prefix_raw",
            ),
            call(
                "./visibilities/ms_prefix/modelvis.ps.zarr",
                upstream_output["modelvis"],
                "ms_prefix_modelvis",
            ),
            call(
                "./visibilities/ms_prefix/raw_1.ps.zarr",
                upstream_output["vis"],
                "ms_prefix_raw_1",
            ),
            call(
                "./visibilities/ms_prefix/modelvis_1.ps.zarr",
                upstream_output["modelvis"],
                "ms_prefix_modelvis_1",
            ),
        ]
    )


def test_should_only_log_when_processing_set_write_completes():
    assert _wait_for_ps_write("out.ps.zarr", None) is None


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
