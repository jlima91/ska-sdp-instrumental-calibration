import numpy as np
import pytest
import xarray as xr
from mock import MagicMock, patch
from ska_sdp_datamodels.calibration import GainTable

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable

MODULE = "ska_sdp_instrumental_calibration.data_managers.data_export"
MODULE += ".export_to_h5parm"


@pytest.fixture
def gaintable(generate_vis) -> GainTable:
    _, gaintable = generate_vis
    return gaintable.copy(deep=True)


@pytest.fixture
def delaytable(gaintable: GainTable) -> DelayTable:
    shape = (1, gaintable.antenna.size, 2)
    return DelayTable.constructor(
        delay=np.zeros(shape),
        offset=np.zeros(shape),
        time=gaintable.time.data[:1],
        antenna=gaintable.antenna.data,
        pol=["XX", "YY"],
        configuration=gaintable.configuration,
    )


def _export_gaintable(gaintable: GainTable, **kwargs) -> xr.Dataset:
    """Export without file I/O, returning the gaintable to be written."""
    with patch(f"{MODULE}.h5py"), patch(
        f"{MODULE}.create_soltab_datasets",
        return_value=(MagicMock(), MagicMock()),
    ) as mock_datasets:
        export_to_h5parm.export_gaintable_to_h5parm(
            gaintable, "unused", **kwargs
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


def test_should_raise_exception_for_unexpected_gaintable_dims(gaintable):
    gaintable = gaintable.transpose("antenna", "time", ...)

    with pytest.raises(ValueError, match="Unexpected dims:"):
        export_to_h5parm.export_gaintable_to_h5parm(gaintable, "unused")


def test_should_raise_exception_for_non_linear_gaintable_pols(gaintable):
    gaintable = gaintable.assign_coords(
        receptor1=["R", "L"], receptor2=["R", "L"]
    )

    with pytest.raises(
        ValueError, match="Subsequent pipelines assume linear pol order"
    ):
        export_to_h5parm.export_gaintable_to_h5parm(gaintable, "unused")


def test_should_raise_exception_for_gaintable_without_configuration(
    gaintable,
):
    gaintable.attrs["configuration"] = None

    with pytest.raises(
        ValueError, match="Missing gt config. H5Parm requires antenna names"
    ):
        export_to_h5parm.export_gaintable_to_h5parm(gaintable, "unused")


def test_should_keep_all_pols_and_axes_of_gaintable_by_default(gaintable):
    gaintable = gaintable.isel(time=[0])

    exported = _export_gaintable(gaintable)

    assert list(exported.gain.sizes) == ["time", "ant", "freq", "pol"]
    assert list(exported.pol.data.astype(str)) == ["XX", "XY", "YX", "YY"]


def test_should_exclude_cross_pols_of_gaintable(gaintable):
    exported = _export_gaintable(gaintable, exclude_cross_pols=True)

    assert list(exported.pol.data.astype(str)) == ["XX", "YY"]


def test_should_squeeze_gaintable(gaintable):
    gaintable = gaintable.isel(time=[0])

    exported = _export_gaintable(gaintable, squeeze=True)

    assert list(exported.gain.sizes) == ["ant", "freq", "pol"]


def test_should_raise_exception_for_unexpected_delaytable_dims(delaytable):
    delaytable = delaytable.transpose("antenna", "time", "pol")

    with pytest.raises(ValueError, match="Unexpected dims:"):
        export_to_h5parm.export_clock_to_h5parm(delaytable, "unused")


def test_should_raise_exception_for_non_linear_delaytable_pols(delaytable):
    delaytable = delaytable.assign_coords(pol=["RR", "LL"])

    with pytest.raises(
        ValueError, match="Subsequent pipelines assume linear pol order"
    ):
        export_to_h5parm.export_clock_to_h5parm(delaytable, "unused")


def test_should_raise_exception_for_delaytable_without_configuration(
    delaytable,
):
    delaytable.attrs["configuration"] = None

    with pytest.raises(
        ValueError, match="Missing gt config. H5Parm requires antenna names"
    ):
        export_to_h5parm.export_clock_to_h5parm(delaytable, "unused")


def test_should_keep_all_axes_of_delaytable_by_default(delaytable):
    exported = _export_delaytable(delaytable)

    assert list(exported.delay.sizes) == ["time", "ant", "pol"]


def test_should_squeeze_delaytable(delaytable):
    exported = _export_delaytable(delaytable, squeeze=True)

    assert list(exported.delay.sizes) == ["ant", "pol"]
