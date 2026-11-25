import h5py
import numpy as np
import pytest
from mock import patch

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_gaintable_to_h5parm,
)
from ska_sdp_instrumental_calibration.data_managers.schema.calibration_set import (  # noqa: E501
    GainCalibrationSetXds,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.service import (
    gaintable,
    read_gaintable,
)

STATION_NAMES = ["station_a", "station_b"]
POLARIZATIONS = ["XX", "XY", "YX", "YY"]
CREATE_GAINTABLE_FROM_H5PARM = (
    "ska_sdp_instrumental_calibration.qa_metrics.service.gaintable"
    ".create_gaintable_from_h5parm"
)


@pytest.fixture(autouse=True)
def clear_gaintable_cache():
    """
    A gaintable read cached by one test must not leak into the next
    """
    # pylint: disable-next=protected-access
    gaintable._read_one.cache_clear()
    yield
    # pylint: disable-next=protected-access
    gaintable._read_one.cache_clear()


def _write_gaintable(path, time, station_names=STATION_NAMES, nchannels=2):
    """
    Write a gaintable as the pipeline does, with unit gains and weights
    """
    shape = (len(time), len(station_names), nchannels, len(POLARIZATIONS))
    export_gaintable_to_h5parm(
        GainCalibrationSetXds.constructor(
            gain=np.ones(shape, dtype=complex),
            weight=np.ones(shape),
            residual=np.zeros((len(time), nchannels, len(POLARIZATIONS))),
            time=np.asarray(time, dtype=float),
            interval=np.ones(len(time)),
            frequency=np.linspace(1e8, 2e8, nchannels),
            antenna_name=np.array(station_names),
            polarization=np.array(POLARIZATIONS),
            jones_type="B",
        ),
        str(path),
    )
    return str(path)


def test_should_read_gaintable_written_by_export(tmp_path):
    path = _write_gaintable(tmp_path / "a.h5parm", [10.0, 20.0])

    actual = read_gaintable([path])

    assert dict(actual.sizes) == {
        "time": 2,
        "antenna_name": 2,
        "frequency": 2,
        "polarization": 4,
    }
    assert list(actual.antenna_name.values) == STATION_NAMES
    assert list(actual.polarization.values) == POLARIZATIONS
    np.testing.assert_array_equal(actual.time, [10.0, 20.0])
    # The solution interval is not stored in the H5parm
    assert np.isnan(actual.CALPARAM_INTERVAL.values).all()


def test_should_read_a_gaintable_once_for_the_same_path(tmp_path):
    path = _write_gaintable(tmp_path / "a.h5parm", [10.0])

    with patch(
        CREATE_GAINTABLE_FROM_H5PARM,
        wraps=gaintable.create_gaintable_from_h5parm,
    ) as create_mock:
        first = read_gaintable([path])
        second = read_gaintable([path])

    # A single gaintable is returned as read, with no concatenation
    assert first is second
    create_mock.assert_called_once()


def test_should_concatenate_gaintables_in_time_order(tmp_path):
    # Gaintables are not given in time order
    paths = [
        _write_gaintable(tmp_path / "b.h5parm", [30.0, 40.0]),
        _write_gaintable(tmp_path / "a.h5parm", [10.0, 20.0]),
    ]

    actual = read_gaintable(paths)

    np.testing.assert_array_equal(actual.time, [10.0, 20.0, 30.0, 40.0])
    assert list(actual.antenna_name.values) == STATION_NAMES


def test_should_raise_if_no_gaintables():
    with pytest.raises(RunMetadataError, match="No gaintable recorded"):
        read_gaintable([])


def test_should_raise_if_gaintable_was_not_produced(tmp_path):
    with pytest.raises(RunMetadataError, match="was not produced"):
        read_gaintable([str(tmp_path / "missing.h5parm")])


def test_should_raise_if_gaintable_is_corrupt(tmp_path):
    path = tmp_path / "corrupt.h5parm"
    path.write_text("not an HDF5 file")

    with pytest.raises(RunMetadataError, match="cannot be read"):
        read_gaintable([str(path)])


def test_should_raise_if_gaintable_has_no_gain_soltab(tmp_path):
    path = tmp_path / "no_soltab.h5parm"
    with h5py.File(path, "w") as h5f:
        h5f.create_group("sol000")

    with pytest.raises(RunMetadataError, match="cannot be read"):
        read_gaintable([str(path)])


def test_should_raise_if_gaintables_have_different_stations(tmp_path):
    paths = [
        _write_gaintable(tmp_path / "a.h5parm", [10.0]),
        _write_gaintable(
            tmp_path / "b.h5parm",
            [20.0],
            station_names=["station_a", "station_c"],
        ),
    ]

    with pytest.raises(RunMetadataError, match="not have the same stations"):
        read_gaintable(paths)


def test_should_not_cache_errors(tmp_path):
    # A gaintable that could not be read is read again on the next query
    path = tmp_path / "late.h5parm"

    with pytest.raises(RunMetadataError, match="was not produced"):
        read_gaintable([str(path)])

    _write_gaintable(path, [10.0])

    assert list(read_gaintable([str(path)]).time.values) == [10.0]
