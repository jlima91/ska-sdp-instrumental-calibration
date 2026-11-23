import h5py
import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_clock_to_h5parm,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.products import (
    read_delaytable,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable

STATION_NAMES = ["station_a", "station_b"]


def _write_delaytable(path, time, delay, station_names=STATION_NAMES):
    """
    Write a delay table as the delay stage does, with the delays of shape
    (time, antenna, pol) and offsets of a tenth of them
    """
    delay = np.asarray(delay, dtype=float)
    export_clock_to_h5parm(
        DelayTable.constructor(
            delay=delay,
            offset=delay / 10,
            time=np.asarray(time, dtype=float),
            antenna_name=np.array(station_names),
            pol=["XX", "YY"],
        ),
        str(path),
    )
    return str(path)


def test_should_read_delaytable_written_by_export(tmp_path):
    delay = [[[1e-9, 2e-9], [3e-9, 4e-9]]]
    path = _write_delaytable(tmp_path / "a.h5parm", [100.0], delay)

    delaytable = read_delaytable([path])

    # Station names come back without the null the export appends
    assert list(delaytable.antenna_name.values) == STATION_NAMES
    assert list(delaytable.pol.values) == ["XX", "YY"]
    np.testing.assert_array_equal(delaytable.time, [100.0])
    np.testing.assert_allclose(delaytable.delay, delay)
    np.testing.assert_allclose(delaytable.offset, np.asarray(delay) / 10)


def test_should_concatenate_delaytables_of_all_ms_in_time_order(tmp_path):
    # MSs are not recorded in time order
    paths = [
        _write_delaytable(
            tmp_path / "b.h5parm", [200.0], [[[2e-9, 2e-9], [2e-9, 2e-9]]]
        ),
        _write_delaytable(
            tmp_path / "a.h5parm", [100.0], [[[1e-9, 1e-9], [1e-9, 1e-9]]]
        ),
    ]

    delaytable = read_delaytable(paths)

    np.testing.assert_array_equal(delaytable.time, [100.0, 200.0])
    np.testing.assert_allclose(
        delaytable.delay.sel(pol="XX", antenna_name="station_a"),
        [1e-9, 2e-9],
    )


def test_should_raise_if_no_delaytables():
    with pytest.raises(RunMetadataError, match="No delay table recorded"):
        read_delaytable([])


def test_should_raise_if_delaytable_was_not_produced(tmp_path):
    with pytest.raises(RunMetadataError, match="was not produced"):
        read_delaytable([str(tmp_path / "missing.h5parm")])


def test_should_raise_if_delaytable_is_corrupt(tmp_path):
    path = tmp_path / "corrupt.h5parm"
    path.write_text("not an HDF5 file")

    with pytest.raises(RunMetadataError, match="cannot be read"):
        read_delaytable([str(path)])


def test_should_raise_if_delaytable_has_no_clock_soltab(tmp_path):
    path = tmp_path / "no_clock.h5parm"
    with h5py.File(path, "w") as h5f:
        h5f.create_group("sol000")

    with pytest.raises(RunMetadataError, match="cannot be read"):
        read_delaytable([str(path)])


def test_should_raise_if_delaytable_axes_are_not_time_ant_pol(tmp_path):
    path = _write_delaytable(
        tmp_path / "a.h5parm", [100.0], [[[1e-9, 1e-9], [1e-9, 1e-9]]]
    )
    with h5py.File(path, "r+") as h5f:
        h5f["sol000/clock000/val"].attrs["AXES"] = np.bytes_("ant,time,pol")

    with pytest.raises(RunMetadataError, match="has axes ant,time,pol"):
        read_delaytable([path])


def test_should_raise_if_delaytables_have_different_stations(tmp_path):
    paths = [
        _write_delaytable(
            tmp_path / "a.h5parm", [100.0], [[[1e-9, 1e-9], [1e-9, 1e-9]]]
        ),
        _write_delaytable(
            tmp_path / "b.h5parm",
            [200.0],
            [[[1e-9, 1e-9], [1e-9, 1e-9]]],
            station_names=["station_a", "station_c"],
        ),
    ]

    with pytest.raises(RunMetadataError, match="not have the same stations"):
        read_delaytable(paths)
