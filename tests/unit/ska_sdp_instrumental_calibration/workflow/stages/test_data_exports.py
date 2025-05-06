import os

from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.workflow.stages.data_exports import (
    export_gaintable_stage,
)

INST_METADATA_FILE = "ska-data-product.yaml"


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
        export_metadata=False,
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
        export_metadata=False,
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


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".INSTMetaData"
)
@patch("ska_sdp_instrumental_calibration.workflow.stages.data_exports" ".dask")
def test_should_export_metadata(
    dask_mock, inst_metadata_mock, export_gaintable_h5parm_mock
):
    inst_metadata_mock.return_value = inst_metadata_mock
    inst_metadata_mock.can_create_metadata.return_value = True
    upstream_output_mock = Mock(name="upstream output")
    upstream_output_mock.__setitem__ = Mock(name="upstream-output-setitem")
    upstream_output_mock["gaintable"] = Mock(name="gaintable")
    dataproduct_mock = Mock(name="dataproducts")
    dataproduct_mock.return_value = [
        {"dp_path": "test_gains.h5parm", "description": "Gaintable"}
    ]

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    export_gaintable_stage.stage_definition(
        upstream_output_mock,
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=True,
        _output_dir_="dir/to/save",
    )

    expected_path = os.path.join("dir/to/save", INST_METADATA_FILE)

    inst_metadata_mock.assert_called_once_with(
        expected_path, data_products=dataproduct_mock.return_value
    )
    upstream_output_mock.add_compute_tasks.assert_has_calls
    ([call(export_gaintable_h5parm_mock()), call(inst_metadata_mock.export())])


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.data_exports"
    ".INSTMetaData"
)
@patch("ska_sdp_instrumental_calibration.workflow.stages.data_exports" ".dask")
def test_should_not_export_metadata_if_prerequisites_are_not_met(
    dask_mock, inst_metadata_mock, export_gaintable_h5parm_mock
):
    inst_metadata_mock.can_create_metadata.return_value = False
    upstream_output_mock = Mock(name="upstream output")
    upstream_output_mock.__setitem__ = Mock(name="upstream-output-setitem")
    upstream_output_mock["gaintable"] = Mock(name="gaintable")
    dataproduct_mock = Mock(name="dataproducts")
    dataproduct_mock.return_value = [
        {"dp_path": "test_gains.h5parm", "description": "Gaintable"}
    ]

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    export_gaintable_stage.stage_definition(
        upstream_output_mock,
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=True,
        _output_dir_="dir/to/save",
    )

    upstream_output_mock.add_compute_tasks.assert_called_once_with(
        export_gaintable_h5parm_mock()
    )
