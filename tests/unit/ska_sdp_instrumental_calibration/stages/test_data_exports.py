from mock import Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages.data_exports import (
    INST_METADATA_FILE,
    concat_gaintables,
    export_gaintable_stage,
    export_gaintable_to_hdf5,
)
from ska_sdp_instrumental_calibration.tagger import Tags


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "export_gain_table": {
            "file_name": "gaintable",
            "export_format": "h5parm",
            "export_metadata": False,
        },
    }

    assert export_gaintable_stage.__stage__.config == expected_config


def test_export_gaintable_stage_is_required():
    assert export_gaintable_stage.__stage__.is_enabled


def test_export_gaintable_stage_is_an_aggregator():
    assert export_gaintable_stage in Tags.AGGREGATOR


@patch("ska_sdp_instrumental_calibration.stages.data_exports.xr")
def test_should_concat_gaintables_in_upstream_outputs(xarray_mock):
    upstream_output = UpstreamOutput()
    upstream_output.calibration_tables = "gaintable"
    upstream_output["gaintable"] = "gaintable_1"

    upstream_output_1 = UpstreamOutput()
    upstream_output_1["gaintable"] = "gaintable_2"

    upstream_output_2 = UpstreamOutput()
    upstream_output_2["gaintable"] = "gaintable_3"

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
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.delayed",
    side_effect=lambda f: f,
)
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports."
    "get_gaintable_file_path"
)
def test_should_export_gaintable_as_h5parm(
    prepare_model_mock,
    delayed_mock,
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
    prepare_model_mock.side_effect = [
        f"{sdm_path}/field_a/test_gains.h5parm",
        f"{sdm_path}/field_b/test_gains.h5parm",
    ]

    export_gaintable_stage(
        [upstream_output1, upstream_output2, upstream_output3],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=False,
        sdm_path=sdm_path,
    )

    export_gaintable_h5parm_mock.assert_has_calls(
        [
            call(upstream_output1.gaintable, expected_path1, False),
            call(upstream_output3.gaintable, expected_path2, False),
        ]
    )
    delayed_mock.assert_has_calls(
        [
            call(export_gaintable_h5parm_mock),
            call(export_gaintable_h5parm_mock),
        ]
    )

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
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.delayed",
    side_effect=lambda f: f,
)
def test_should_export_gaintable_as_hdf5(
    delayed_mock, concat_mock, export_gaintable_hdf5_mock
):

    upstream_output = _get_prepopulated_upstream_output()
    concat_mock.return_value = upstream_output
    export_gaintable_hdf5_mock.return_value = "field_a_export"

    export_gaintable_stage(
        [upstream_output],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="hdf5",
        export_metadata=False,
    )

    export_gaintable_hdf5_mock.assert_called_once_with(
        upstream_output.gaintable, "dir/to/save/field_a_test_gains.hdf5", False
    )
    delayed_mock.assert_called_once_with(export_gaintable_hdf5_mock)


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.INSTMetaData")
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.concat_gaintables"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.delayed",
    side_effect=lambda f: f,
)
def test_should_export_metadata(
    delayed_mock, concat_mock, inst_metadata_mock, export_gaintable_h5parm_mock
):
    inst_metadata_mock.return_value = inst_metadata_mock
    inst_metadata_mock.can_create_metadata.return_value = True
    upstream_output = _get_prepopulated_upstream_output()
    concat_mock.return_value = upstream_output
    dataproduct_mock = Mock(name="dataproducts")
    dataproduct_mock.return_value = [
        {"dp_path": "field_a_test_gains.h5parm", "description": "Gaintable"}
    ]

    export_gaintable_stage(
        [upstream_output],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=True,
    )

    inst_metadata_mock.assert_called_once_with(
        f"dir/to/save/{INST_METADATA_FILE}",
        data_products=dataproduct_mock.return_value,
    )
    export_gaintable_h5parm_mock.assert_called_once_with(
        upstream_output["gaintable"],
        "dir/to/save/field_a_test_gains.h5parm",
        False,
    )
    inst_metadata_mock.export.assert_called_once()


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports"
    ".export_gaintable_to_h5parm"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.INSTMetaData")
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.concat_gaintables"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports.delayed",
    side_effect=lambda f: f,
)
def test_should_not_export_metadata_if_prerequisites_are_not_met(
    delayed_mock, concat_mock, inst_metadata_mock, export_gaintable_h5parm_mock
):
    inst_metadata_mock.can_create_metadata.return_value = False
    upstream_output = _get_prepopulated_upstream_output()
    concat_mock.return_value = upstream_output
    dataproduct_mock = Mock(name="dataproducts")
    dataproduct_mock.return_value = [
        {"dp_path": "test_gains.h5parm", "description": "Gaintable"}
    ]

    export_gaintable_stage(
        [upstream_output],
        _output_dir_="dir/to/save",
        file_name="test_gains",
        export_format="h5parm",
        export_metadata=True,
    )

    export_gaintable_h5parm_mock.assert_called_once_with(
        upstream_output["gaintable"],
        "dir/to/save/field_a_test_gains.h5parm",
        False,
    )
    inst_metadata_mock.assert_not_called()


def _get_prepopulated_upstream_output(
    field_id="field_a", calibration_purpose="gains"
):
    upstream_output = UpstreamOutput()
    upstream_output["gaintable"] = Mock(name="gaintable")
    upstream_output["field_id"] = field_id
    upstream_output["calibration_purpose"] = calibration_purpose
    upstream_output.calibration_tables = "gaintable"
    return upstream_output


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports."
    "convert_gaintable_to_hdf"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.h5py.File")
def test_export_gaintable_to_hdf5(
    h5py_file_mock,
    convert_mock,
):
    """Test GainTable export without polarization filtering."""

    gaintable_mock = Mock(name="gaintable")

    file_handle_mock = Mock(name="file_handle")
    file_handle_mock.attrs = {}
    group_mock = Mock(name="group")

    h5py_file_mock.return_value.__enter__.return_value = file_handle_mock
    file_handle_mock.create_group.return_value = group_mock

    export_gaintable_to_hdf5(
        gaintable_mock,
        "test.hdf5",
        exclude_cross_pols=False,
    )

    h5py_file_mock.assert_called_once_with(
        "test.hdf5",
        "w",
    )

    file_handle_mock.create_group.assert_called_once_with("GainTable0")

    convert_mock.assert_called_once_with(
        gaintable_mock,
        group_mock,
    )

    file_handle_mock.flush.assert_called_once()


@patch(
    "ska_sdp_instrumental_calibration.stages.data_exports."
    "convert_gaintable_to_hdf"
)
@patch("ska_sdp_instrumental_calibration.stages.data_exports.h5py.File")
def test_export_gaintable_to_hdf5_exclude_cross_pols(
    h5py_file_mock,
    convert_mock,
):
    """Test GainTable export with XY/YX removed."""

    filtered_gaintable_mock = Mock(name="filtered_gaintable")

    gaintable_mock = Mock(name="gaintable")
    gaintable_mock.where.return_value = filtered_gaintable_mock

    file_handle_mock = Mock(name="file_handle")
    file_handle_mock.attrs = {}
    group_mock = Mock(name="group")

    h5py_file_mock.return_value.__enter__.return_value = file_handle_mock
    file_handle_mock.create_group.return_value = group_mock

    export_gaintable_to_hdf5(
        gaintable_mock,
        "test.hdf5",
        exclude_cross_pols=True,
    )

    gaintable_mock.where.assert_called_once_with(
        gaintable_mock.receptor1 == gaintable_mock.receptor2,
        drop=True,
    )

    convert_mock.assert_called_once_with(
        filtered_gaintable_mock,
        group_mock,
    )
