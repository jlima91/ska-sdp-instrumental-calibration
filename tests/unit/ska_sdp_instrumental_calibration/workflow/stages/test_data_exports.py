import os

from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages.data_exports import (
    export_gaintable_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
def test_should_export_gaintable_as_h5parm(export_gaintable_h5parm_mock):

    upstream_output = UpstreamOutput()
    upstream_output["gaintable"] = Mock(name="gaintable")

    actual_output = export_gaintable_stage.stage_definition(
        upstream_output,
        file_name="test_gains",
        export_format="h5parm",
        _output_dir_="dir/to/save",
    )

    expected_path = os.path.join("dir/to/save", "test_gains.h5parm")

    export_gaintable_h5parm_mock.assert_called_once_with(
        upstream_output.gaintable,
        expected_path,
    )

    assert actual_output == upstream_output


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".export_gaintable_to_hdf5"
)
def test_should_export_gaintable_as_hdf5(export_gaintable_hdf5_mock):

    upstream_output = UpstreamOutput()
    upstream_output["gaintable"] = Mock(name="gaintable")

    actual_output = export_gaintable_stage.stage_definition(
        upstream_output,
        file_name="test_gains",
        export_format="hdf5",
        _output_dir_="dir/to/save",
    )

    expected_path = os.path.join("dir/to/save", "test_gains.hdf5")

    export_gaintable_hdf5_mock.assert_called_once_with(
        upstream_output.gaintable,
        expected_path,
    )

    assert actual_output == upstream_output
