import os

from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.data_exports import (
    concat_gaintables,
    export_gaintable_stage,
)
from ska_sdp_instrumental_calibration.tagger import Tags

INST_METADATA_FILE = "ska-data-product.yaml"


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "export_gain_table": {
            "file_name": "inst.gaintable",
            "export_format": "h5parm",
            "export_metadata": False,
        },
    }

    assert export_gaintable_stage.__stage__.config == expected_config


def test_export_gaintable_stage_is_required():
    assert export_gaintable_stage.__stage__.is_enabled


def test_export_gaintable_stage_is_an_aggregator():
    assert export_gaintable_stage in Tags.AGGREGATOR


def test_should_not_concat_gaintables_in_upstream_outputs():
    upstream_output = Mock(name="upstream_output")
    upstream_output.gaintable = "gaintable_1"

    result = concat_gaintables([upstream_output])

    assert result == upstream_output


@patch("ska_sdp_instrumental_calibration.stages.data_exports.xr")
def test_should_concat_gaintables_in_upstream_outputs(xarray_mock):
    upstream_output = Mock(name="upstream_output")
    upstream_output.gaintable = "gaintable_1"

    upstream_output_1 = Mock(name="ugpstream_output_1")
    upstream_output_1.gaintable = "gaintable_2"

    upstream_output_2 = Mock(name="upstream_output_2")
    upstream_output_2.gaintable = "gaintable_3"

    result = concat_gaintables(
        [upstream_output, upstream_output_1, upstream_output_2]
    )

    xarray_mock.concat.assert_called_once_with(
        ["gaintable_1", "gaintable_2", "gaintable_3"], dim="time"
    )

    assert result.gaintable == xarray_mock.concat.return_value
    assert upstream_output.gaintable == xarray_mock.concat.return_value


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.concat_gaintables"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.dask")
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports."
    "get_gaintable_file_path"
)
def test_should_export_gaintable_as_h5parm(
    prepare_model_mock,
    dask_mock,
    concat_mock,
    export_gaintable_h5parm_mock,
):

    sdm_path = "/path/to/sdm"
    expected_path1 = "/path/to/sdm/field_a/test_gains.h5parm"
    expected_path2 = "/path/to/sdm/field_b/test_gains.h5parm"

    upstream_output1 = _get_prepopulated_upstream_output(field_id="field_a")
    upstream_output2 = _get_prepopulated_upstream_output(field_id="field_a")
    upstream_output3 = _get_prepopulated_upstream_output(field_id="field_b")

    concat_mock.side_effect = [upstream_output1, upstream_output3]
    export_gaintable_h5parm_mock.side_effect = [
        "field_a_export",
        "field_b_export",
    ]
    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock
    prepare_model_mock.side_effect = [
        f"{sdm_path}/field_a/test_gains.h5parm",
        f"{sdm_path}/field_b/test_gains.h5parm",
    ]

    actual_output = export_gaintable_stage(
        [upstream_output1, upstream_output2, upstream_output3],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=False,
        sdm_path=sdm_path,
    )

    export_gaintable_h5parm_mock.assert_has_calls(
        [
            call(upstream_output1.gaintable, expected_path1),
            call(upstream_output3.gaintable, expected_path2),
        ]
    )
    dask_mock.delayed.assert_has_calls(
        [
            call(export_gaintable_h5parm_mock),
            call(export_gaintable_h5parm_mock),
        ]
    )

    assert "field_a_export" in actual_output.compute_tasks
    assert "field_b_export" in actual_output.compute_tasks
    concat_mock.assert_has_calls(
        [
            call([upstream_output1, upstream_output2]),
            call([upstream_output3]),
        ]
    )

    prepare_model_mock.assert_has_calls(
        [
            call(
                output_dir="dir/to/save",
                filename="test_gains.h5parm",
                sdm_path="/path/to/sdm",
                purpose="gains",
                field_id="field_a",
            ),
            call(
                output_dir="dir/to/save",
                filename="test_gains.h5parm",
                sdm_path="/path/to/sdm",
                purpose="gains",
                field_id="field_b",
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports"
    ".export_gaintable_to_hdf5"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.concat_gaintables"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.dask")
def test_should_export_gaintable_as_hdf5(
    dask_mock, concat_mock, export_gaintable_hdf5_mock
):

    upstream_output = _get_prepopulated_upstream_output()
    concat_mock.return_value = upstream_output
    export_gaintable_hdf5_mock.return_value = "field_a_export"

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    actual_output = export_gaintable_stage(
        [upstream_output],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="hdf5",
        export_metadata=False,
    )

    expected_path = os.path.join("dir/to/save", "field_a_test_gains.hdf5")

    export_gaintable_hdf5_mock.assert_called_once_with(
        upstream_output.gaintable, expected_path
    )
    dask_mock.delayed.assert_called_once_with(export_gaintable_hdf5_mock)
    assert "field_a_export" in actual_output.compute_tasks


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.INSTMetaData")
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.concat_gaintables"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.dask")
def test_should_export_metadata(
    dask_mock, concat_mock, inst_metadata_mock, export_gaintable_h5parm_mock
):
    inst_metadata_mock.return_value = inst_metadata_mock
    inst_metadata_mock.can_create_metadata.return_value = True
    upstream_output = _get_prepopulated_upstream_output()
    concat_mock.return_value = upstream_output
    dataproduct_mock = Mock(name="dataproducts")
    dataproduct_mock.return_value = [
        {"dp_path": "test_gains.h5parm", "description": "Gaintable"}
    ]

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    actual_output = export_gaintable_stage(
        [upstream_output],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=True,
    )

    expected_path = os.path.join("dir/to/save", INST_METADATA_FILE)

    inst_metadata_mock.assert_called_once_with(
        expected_path, data_products=dataproduct_mock.return_value
    )
    assert export_gaintable_h5parm_mock() in actual_output.compute_tasks
    assert inst_metadata_mock.export() in actual_output.compute_tasks


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.INSTMetaData")
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.concat_gaintables"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.dask")
def test_should_not_export_metadata_if_prerequisites_are_not_met(
    dask_mock, concat_mock, inst_metadata_mock, export_gaintable_h5parm_mock
):
    inst_metadata_mock.can_create_metadata.return_value = False
    upstream_output = _get_prepopulated_upstream_output()
    concat_mock.return_value = upstream_output
    dataproduct_mock = Mock(name="dataproducts")
    dataproduct_mock.return_value = [
        {"dp_path": "test_gains.h5parm", "description": "Gaintable"}
    ]

    delayed_mock = Mock(side_effect=lambda f: f)
    dask_mock.delayed = delayed_mock

    actual_output = export_gaintable_stage(
        [upstream_output],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=True,
    )

    assert len(actual_output.compute_tasks) == 1
    assert export_gaintable_h5parm_mock() in actual_output.compute_tasks
    inst_metadata_mock.assert_not_called()


def _get_prepopulated_upstream_output(
    field_id="field_a", calibration_purpose="gains"
):
    upstream_output = UpstreamOutput()
    upstream_output["gaintable"] = Mock(name="gaintable")
    upstream_output["field_id"] = field_id
    upstream_output["calibration_purpose"] = calibration_purpose
    return upstream_output
