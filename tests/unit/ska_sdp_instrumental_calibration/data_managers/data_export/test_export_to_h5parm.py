import numpy as np
import pytest
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm,
)


def test_should_create_soltab_dataset():
    soltab = MagicMock(name="soltab")
    gaintable = MagicMock(name="gaintbale")
    data_mock = MagicMock(name="data")
    gaintable.__getitem__.return_value = data_mock
    val_mock = MagicMock(name="val")
    weight_mock = MagicMock(name="weight")
    soltab.create_dataset.side_effect = ("A", "B", val_mock, weight_mock)

    gaintable.gain.sizes = ["A", "B"]

    val, weight = export_to_h5parm.create_soltab_datasets(soltab, gaintable)

    soltab.create_dataset.assert_has_calls(
        [
            call("A", data=data_mock.data),
            call("B", data=data_mock.data),
            call("val", shape=gaintable.gain.shape, dtype=float),
            call("weight", shape=gaintable.gain.shape, dtype=float),
        ]
    )

    gaintable.__getitem__.assert_has_calls([call("A"), call("B")])
    val_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    weight_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    assert val == val_mock
    assert weight == weight_mock


def test_should_create_clock_soltab_dataset():
    soltab = MagicMock(name="soltab")
    gaintable = MagicMock(name="gaintbale")
    data_mock = MagicMock(name="data")
    gaintable.__getitem__.return_value = data_mock
    val_mock = MagicMock(name="val")
    offset_mock = MagicMock(name="offset")
    soltab.create_dataset.side_effect = ("A", "B", val_mock, offset_mock)

    gaintable.delay.sizes = ["A", "B"]

    val, offset = export_to_h5parm.create_clock_soltab_datasets(
        soltab, gaintable
    )

    soltab.create_dataset.assert_has_calls(
        [
            call("A", data=data_mock.data),
            call("B", data=data_mock.data),
            call("val", shape=gaintable.delay.shape, dtype=float),
            call("offset", shape=gaintable.delay.shape, dtype=float),
        ]
    )

    gaintable.__getitem__.assert_has_calls([call("A"), call("B")])
    val_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    offset_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    assert val == val_mock
    assert offset == offset_mock


def test_should_create_soltab_group():
    soltab = MagicMock(name="soltab")
    solset = MagicMock(name="solset")
    solset.create_group.return_value = soltab

    soltb = export_to_h5parm.create_soltab_group(solset, "phase")

    solset.create_group.assert_called_once_with("phase000")
    soltab.attrs.__setitem__.assert_called_once_with(
        "TITLE", np.bytes_("phase")
    )

    assert soltb == soltab


@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.np"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.h5py"
)
def test_should_raise_exceptions(h5py_mock, np_mock):
    gaintable_mock = MagicMock(name="gaintable")
    with pytest.raises(ValueError, match=r"Unexpected dims:"):
        export_to_h5parm.export_gaintable_to_h5parm(gaintable_mock, "filename")

    gaintable_mock.gain.sizes = [
        "time",
        "antenna",
        "frequency",
        "receptor1",
        "receptor2",
    ]
    np_mock.array_equal.return_value = False

    with pytest.raises(
        ValueError, match="Subsequent pipelines assume linear pol order"
    ):
        export_to_h5parm.export_gaintable_to_h5parm(gaintable_mock, "filename")

    np_mock.array_equal.return_value = True
    gaintable_mock.configuration = None
    gaintable_mock.assign_coords.return_value = gaintable_mock
    gaintable_mock.isel.return_value = gaintable_mock
    gaintable_mock.rename.return_value = gaintable_mock
    gaintable_mock.stack.return_value = gaintable_mock

    with pytest.raises(
        ValueError, match="Missing gt config. H5Parm requires antenna names"
    ):
        export_to_h5parm.export_gaintable_to_h5parm(gaintable_mock, "filename")


@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.create_soltab_datasets"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.create_soltab_group"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.np"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.h5py"
)
def test_should_export_gaintable_to_h5parm(
    h5py_mock, np_mock, mock_soltab_group, mock_soltab_dataset
):
    gaintable_mock = MagicMock(name="gaintable")
    stacked_gaintable_mock = MagicMock(name="stack_gaintable")
    mock_file = MagicMock(name="file")
    mock_solset = MagicMock(name="solset")
    mock_file.create_group.return_value = mock_solset
    h5py_mock.File.return_value.__enter__.return_value = mock_file

    mock_val = MagicMock(name="val")
    mock_weight = MagicMock(name="weight")

    mock_soltab_dataset.return_value = [mock_val, mock_weight]

    gaintable_mock.rename.return_value = gaintable_mock
    np_mock.asarray.return_value = "assarray"

    gaintable_mock.gain.sizes = [
        "time",
        "antenna",
        "frequency",
        "receptor1",
        "receptor2",
    ]

    gaintable_mock.stack.return_value = stacked_gaintable_mock
    stacked_gaintable_mock.assign_coords.return_value = stacked_gaintable_mock
    stacked_gaintable_mock.isel.return_value = stacked_gaintable_mock
    stacked_gaintable_mock.squeeze.return_value = stacked_gaintable_mock

    export_to_h5parm.export_gaintable_to_h5parm(
        gaintable_mock, "filename", squeeze=True
    )
    gaintable_mock.rename.assert_called_once_with(
        {"antenna": "ant", "frequency": "freq"}
    )
    gaintable_mock.stack.assert_called_once_with(
        pol=("receptor1", "receptor2")
    )

    mock_soltab_group.assert_has_calls(
        [call(mock_solset, "amplitude"), call(mock_solset, "phase")]
    )

    mock_soltab_dataset.assert_has_calls(
        [
            call(mock_soltab_group.return_value, stacked_gaintable_mock),
            call(mock_soltab_group.return_value, stacked_gaintable_mock),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.np"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.h5py"
)
def test_should_raise_exceptions_for_clock(h5py_mock, np_mock):
    delaytable_mock = MagicMock(name="gaintable")
    delaytable_mock.delay.sizes = ["time"]
    with pytest.raises(ValueError, match=r"Unexpected dims:"):
        delayed_export = export_to_h5parm.export_clock_to_h5parm(
            delaytable_mock, "filename"
        )
        delayed_export.compute()

    delaytable_mock.delay.sizes = ["time", "antenna", "pol"]
    np_mock.array_equal.return_value = False
    delaytable_mock.rename.return_value = delaytable_mock

    with pytest.raises(
        ValueError, match="Subsequent pipelines assume linear pol order"
    ):
        delayed_export = export_to_h5parm.export_clock_to_h5parm(
            delaytable_mock, "filename"
        )
        delayed_export.compute()

    np_mock.array_equal.return_value = True
    delaytable_mock.configuration = None
    delaytable_mock.assign_coords.return_value = delaytable_mock

    with pytest.raises(
        ValueError, match="Missing gt config. H5Parm requires antenna names"
    ):
        delayed_export = export_to_h5parm.export_clock_to_h5parm(
            delaytable_mock, "filename"
        )
        delayed_export.compute()


@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.create_clock_soltab_datasets"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.create_soltab_group"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.np"
)
@patch(
    "ska_sdp_instrumental_calibration.data_managers."
    "data_export.export_to_h5parm.h5py"
)
def test_should_export_clock_to_h5parm(
    h5py_mock, np_mock, mock_soltab_group, mock_soltab_dataset
):
    delaytable_mock = MagicMock(name="gaintable")
    renamed_delaytable_mock = MagicMock(name="gaintable")
    mock_file = MagicMock(name="file")
    mock_solset = MagicMock(name="solset")
    mock_file.create_group.return_value = mock_solset
    h5py_mock.File.return_value.__enter__.return_value = mock_file

    mock_val = MagicMock(name="val")
    mock_weight = MagicMock(name="weight")

    mock_soltab_dataset.return_value = [mock_val, mock_weight]

    delaytable_mock.rename.return_value = renamed_delaytable_mock
    np_mock.asarray.return_value = "assarray"

    delaytable_mock.delay.sizes = ["time", "antenna", "pol"]

    renamed_delaytable_mock.assign_coords.return_value = (
        renamed_delaytable_mock
    )
    renamed_delaytable_mock.squeeze.return_value = renamed_delaytable_mock

    delayed_call = export_to_h5parm.export_clock_to_h5parm(
        delaytable_mock, "filename", squeeze=True
    )
    delayed_call.compute()

    delaytable_mock.rename.assert_called_once_with({"antenna": "ant"})
    renamed_delaytable_mock.assign_coords.assert_has_calls(
        [
            call({"pol": "assarray"}),
            call({"ant": "assarray"}),
        ]
    )

    mock_soltab_group.assert_called_once_with(mock_solset, "clock")

    mock_soltab_dataset.assert_called_once_with(
        mock_soltab_group.return_value, renamed_delaytable_mock
    )
