import h5py
import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_clock_to_h5parm,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.field_metadata import (  # noqa: E501
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.service import read_delaytable
from ska_sdp_instrumental_calibration.qa_metrics.service.cable_delays import (  # noqa: E501
    get_cable_delays,
    get_delaytable_paths,
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


def _delaytable(delay):
    delay = np.asarray(delay, dtype=float)
    return DelayTable.constructor(
        delay=delay,
        offset=np.zeros_like(delay),
        time=np.arange(delay.shape[0], dtype=float),
        antenna_name=np.array(["station_a", "station_b"]),
        pol=["XX", "YY"],
    )


def test_should_take_median_delay_over_time_in_nanoseconds():
    # (time, antenna, pol): 3 times, 2 stations, XX and YY in seconds
    delaytable = _delaytable(
        [
            [[10e-9, 30e-9], [1e-9, 2e-9]],
            [[20e-9, 40e-9], [3e-9, 4e-9]],
            [[30e-9, 50e-9], [5e-9, 6e-9]],
        ]
    )

    cable_delays = get_cable_delays(delaytable)

    assert cable_delays.values_per_station == {
        "station_a": {
            "x_delay": pytest.approx(20.0),
            "y_delay": pytest.approx(40.0),
        },
        "station_b": {
            "x_delay": pytest.approx(3.0),
            "y_delay": pytest.approx(4.0),
        },
    }


def test_should_not_be_pulled_by_one_badly_solved_scan():
    # station_a: one scan is off by 90 ns; the mean would be 40 ns
    delaytable = _delaytable(
        [
            [[10e-9, 10e-9], [1e-9, 1e-9]],
            [[10e-9, 10e-9], [1e-9, 1e-9]],
            [[100e-9, 10e-9], [1e-9, 1e-9]],
        ]
    )

    values = get_cable_delays(delaytable).values_per_station

    assert values["station_a"]["x_delay"] == pytest.approx(10.0)


def test_should_leave_unsolved_delays_out_of_the_median():
    delaytable = _delaytable(
        [
            [[10e-9, np.nan], [1e-9, 1e-9]],
            [[np.nan, 40e-9], [1e-9, 1e-9]],
            [[30e-9, 60e-9], [1e-9, 1e-9]],
        ]
    )

    values = get_cable_delays(delaytable).values_per_station

    # Medians of (10, 30) and (40, 60), the NaN left out
    assert values["station_a"]["x_delay"] == pytest.approx(20.0)
    assert values["station_a"]["y_delay"] == pytest.approx(50.0)


def test_should_get_delaytable_of_each_ms():
    field_metadata = FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "delay": {
                "delaytable": {
                    "ms_a": "/qa/dir/ms_a-delay.h5parm",
                    "ms_b": "/qa/dir/ms_b-delay.h5parm",
                }
            }
        },
    )

    assert get_delaytable_paths(field_metadata) == [
        "/qa/dir/ms_a-delay.h5parm",
        "/qa/dir/ms_b-delay.h5parm",
    ]


def test_should_get_no_delaytable_if_none_is_recorded():
    # Not an error here: the reader raises for an empty list
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    assert get_delaytable_paths(field_metadata) == []
