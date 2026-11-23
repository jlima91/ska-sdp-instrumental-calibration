import pytest
from mock import Mock, patch

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.products import read_gaintable

READ_H5PARM_GAINS = (
    "ska_sdp_instrumental_calibration.qa_metrics.products.gaintable"
    ".read_h5parm_gains"
)
GAINTABLE = "/qa/dir/field_a.h5parm"


@pytest.fixture(autouse=True)
def clear_gaintable_cache():
    """
    Tests mock read_h5parm_gains differently for the same gaintable path,
    so a gaintable read cached by one test must not leak into the next
    """
    read_gaintable.cache_clear()
    yield
    read_gaintable.cache_clear()


def _gains(antenna):
    return "gain", "weight", "solved", Mock(name="soltab", antenna=antenna)


@patch(READ_H5PARM_GAINS)
def test_should_read_gaintable_once_for_the_same_path(read_h5parm_gains_mock):
    gains = _gains(["station_a"])
    read_h5parm_gains_mock.return_value = gains

    assert read_gaintable(GAINTABLE) is gains
    assert read_gaintable(GAINTABLE) is gains
    read_h5parm_gains_mock.assert_called_once_with(GAINTABLE)


@patch(READ_H5PARM_GAINS)
def test_should_raise_if_gaintable_was_not_produced(read_h5parm_gains_mock):
    read_h5parm_gains_mock.side_effect = FileNotFoundError(GAINTABLE)

    with pytest.raises(RunMetadataError, match="was not produced"):
        read_gaintable(GAINTABLE)


@pytest.mark.parametrize(
    "error",
    [
        # Truncated or corrupt HDF5 file
        OSError("file signature not found"),
        # No solset
        KeyError("sol000"),
        # No amplitude or phase soltab
        ValueError("No amplitude000 or phase000 soltab found"),
    ],
)
@patch(READ_H5PARM_GAINS)
def test_should_raise_if_gaintable_cannot_be_read(
    read_h5parm_gains_mock, error
):
    read_h5parm_gains_mock.side_effect = error

    with pytest.raises(RunMetadataError, match="cannot be read") as raised:
        read_gaintable(GAINTABLE)

    assert raised.value.__cause__ is error


@patch(READ_H5PARM_GAINS)
def test_should_raise_if_gaintable_has_no_antenna_axis(read_h5parm_gains_mock):
    read_h5parm_gains_mock.return_value = _gains(None)

    with pytest.raises(RunMetadataError, match="no antenna axis"):
        read_gaintable(GAINTABLE)


@patch(READ_H5PARM_GAINS)
def test_should_not_cache_errors(read_h5parm_gains_mock):
    # A gaintable that could not be read is read again on the next query
    gains = _gains(["station_a"])
    read_h5parm_gains_mock.side_effect = [OSError("truncated"), gains]

    with pytest.raises(RunMetadataError):
        read_gaintable(GAINTABLE)

    assert read_gaintable(GAINTABLE) is gains
    assert read_h5parm_gains_mock.call_count == 2
