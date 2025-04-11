import os

from mock import Mock, patch

from ska_sdp_instrumental_calibration.workflow.stages.data_exports import (
    export_gaintable_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.workflow.stages.data_exports" ".dask")
def test_should_export_gaintable_as_h5parm(
    dask_mock, export_gaintable_h5parm_mock
):

    upstream_output_mock = Mock(name="upstream output")
    upstream_output_mock.__setitem__ = Mock(name="upstream-output-setitem")
    upstream_output_mock["gaintable"] = Mock(name="gaintable")

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    actual_output = export_gaintable_stage.stage_definition(
        upstream_output_mock,
        file_name="test_gains",
        export_format="h5parm",
        _output_dir_="dir/to/save",
    )

    expected_path = os.path.join("dir/to/save", "test_gains.h5parm")
    export_gaintable_h5parm_mock.assert_called_once_with(
        upstream_output_mock.gaintable, expected_path
    )
    dask_mock.delayed.assert_called_once_with(export_gaintable_h5parm_mock)
    upstream_output_mock.add_compute_tasks.assert_called_once_with(
        export_gaintable_h5parm_mock()
    )
    assert actual_output == upstream_output_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".export_gaintable_to_hdf5"
)
@patch("ska_sdp_instrumental_calibration.workflow.stages.data_exports" ".dask")
def test_should_export_gaintable_as_hdf5(
    dask_mock, export_gaintable_hdf5_mock
):

    upstream_output_mock = Mock(name="upstream output")
    upstream_output_mock.__setitem__ = Mock(name="upstream-output-setitem")
    upstream_output_mock["gaintable"] = Mock(name="gaintable")

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    actual_output = export_gaintable_stage.stage_definition(
        upstream_output_mock,
        file_name="test_gains",
        export_format="hdf5",
        _output_dir_="dir/to/save",
    )

    expected_path = os.path.join("dir/to/save", "test_gains.hdf5")

    export_gaintable_hdf5_mock.assert_called_once_with(
        upstream_output_mock.gaintable, expected_path
    )
    dask_mock.delayed.assert_called_once_with(export_gaintable_hdf5_mock)
    upstream_output_mock.add_compute_tasks.assert_called_once_with(
        export_gaintable_hdf5_mock()
    )
    assert actual_output == upstream_output_mock
