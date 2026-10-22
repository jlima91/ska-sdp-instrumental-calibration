import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable

MODULE = "ska_sdp_instrumental_calibration.data_managers.data_export"
MODULE += ".export_to_h5parm"


@pytest.fixture
def calibration_set(generate_vis) -> xr.Dataset:
    """Return a copy of the schema-based calibration dataset."""
    _, calibration_set = generate_vis
    return calibration_set.copy(deep=True)


@pytest.fixture
def delaytable(calibration_set) -> DelayTable:
    """Build a delay table using the calibration set's antenna names."""
    shape = (1, calibration_set.antenna_name.size, 2)
    return DelayTable.constructor(
        delay=np.zeros(shape),
        offset=np.zeros(shape),
        time=calibration_set.time.data[:1],
        antenna_name=calibration_set.antenna_name.data,
        pol=["XX", "YY"],
    )


def _export_calibration_set(
    calibration_set: xr.Dataset, **kwargs
) -> xr.Dataset:
    """Export without file I/O, returning the dataset passed to H5Parm."""
    with patch(f"{MODULE}.h5py"), patch(
        f"{MODULE}.create_soltab_datasets",
        return_value=(MagicMock(), MagicMock()),
    ) as mock_datasets:
        export_to_h5parm.export_gaintable_to_h5parm(
            calibration_set, "unused", **kwargs
        )
    return mock_datasets.call_args.args[1]


def _export_delaytable(delaytable: DelayTable, **kwargs) -> xr.Dataset:
    """Export without file I/O, returning the delaytable to be written."""
    with patch(f"{MODULE}.h5py"), patch(
        f"{MODULE}.create_clock_soltab_datasets",
        return_value=(MagicMock(), MagicMock()),
    ) as mock_datasets:
        export_to_h5parm.export_clock_to_h5parm(delaytable, "unused", **kwargs)
    return mock_datasets.call_args.args[1]


def test_should_raise_exception_for_unexpected_calibration_set_dims(
    calibration_set,
):
    calibration_set = calibration_set.transpose("antenna_name", "time", ...)

    with pytest.raises(ValueError, match="Unexpected dims:"):
        export_to_h5parm.export_gaintable_to_h5parm(calibration_set, "unused")


def test_should_raise_exception_for_non_linear_calibration_set_pols(
    calibration_set,
):
    calibration_set = calibration_set.assign_coords(
        polarization=["RR", "RL", "LR", "LL"]
    )

    with pytest.raises(
        ValueError, match="Subsequent pipelines assume linear pol order"
    ):
        export_to_h5parm.export_gaintable_to_h5parm(calibration_set, "unused")


def test_should_keep_all_pols_and_axes_of_calibration_set_by_default(
    calibration_set,
):
    calibration_set = calibration_set.isel(time=[0])

    exported = _export_calibration_set(calibration_set)

    assert list(exported.CALPARAM_GAIN.sizes) == ["time", "ant", "freq", "pol"]
    assert list(exported.pol.data.astype(str)) == ["XX", "XY", "YX", "YY"]
    assert list(exported.ant.data.astype(str)) == list(
        calibration_set.antenna_name.data.astype(str)
    )


def test_should_exclude_cross_pols_of_calibration_set(calibration_set):
    exported = _export_calibration_set(
        calibration_set, exclude_cross_pols=True
    )

    assert list(exported.pol.data.astype(str)) == ["XX", "YY"]


def test_should_squeeze_calibration_set(calibration_set):
    calibration_set = calibration_set.isel(time=[0])

    exported = _export_calibration_set(calibration_set, squeeze=True)

    assert list(exported.CALPARAM_GAIN.sizes) == ["ant", "freq", "pol"]


def test_should_raise_exception_for_unexpected_delaytable_dims(delaytable):
    delaytable = delaytable.transpose("antenna_name", "time", "pol")

    with pytest.raises(ValueError, match="Unexpected dims:"):
        export_to_h5parm.export_clock_to_h5parm(delaytable, "unused")


def test_should_raise_exception_for_non_linear_delaytable_pols(delaytable):
    delaytable = delaytable.assign_coords(pol=["RR", "LL"])

    with pytest.raises(
        ValueError, match="Subsequent pipelines assume linear pol order"
    ):
        export_to_h5parm.export_clock_to_h5parm(delaytable, "unused")


def test_should_keep_all_axes_of_delaytable_by_default(delaytable):
    exported = _export_delaytable(delaytable)

    assert list(exported.delay.sizes) == ["time", "ant", "pol"]


def test_should_squeeze_delaytable(delaytable):
    exported = _export_delaytable(delaytable, squeeze=True)

    assert list(exported.delay.sizes) == ["ant", "pol"]
