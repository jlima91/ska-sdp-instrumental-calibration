import dask.array as da
import numpy as np
import pytest
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_msv4,
)
from ska_sdp_instrumental_calibration.xarray_processors.apply import (
    _apply_gaintable_to_dataset_ufunc,
    apply_gaintable_to_dataset,
)


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".apply.apply_antenna_gains_to_visibility"
)
def test_should_apply_gaintable_to_dataset(apply_gaintable_mock):
    vis_mock = MagicMock(name="vis")
    pol_gain_mock = MagicMock(name="pol_gains")
    pol_gain_mock.shape = (2, 4)
    gain_mock = pol_gain_mock.reshape.return_value
    updated_vis_mock = MagicMock(name="applied_vis")
    apply_gaintable_mock.return_value = updated_vis_mock

    gain_mock.__getitem__.return_value = gain_mock
    gain_mock.shape = (2, 2, 2)

    expected = _apply_gaintable_to_dataset_ufunc(
        vis_mock, pol_gain_mock, "antenna1", "antenna2", False
    )

    pol_gain_mock.reshape.assert_called_once_with(2, 2, 2)
    gain_mock.__getitem__.assert_has_calls(
        [call((np.newaxis, ...)), call((np.newaxis, ...))]
    )

    vis_mock.transpose.assert_called_once_with(0, 2, 1, 3)
    gain_mock.transpose.assert_called_once_with(0, 2, 1, 3, 4)

    apply_gaintable_mock.assert_called_once_with(
        vis_mock.transpose.return_value,
        gain_mock.transpose.return_value,
        "antenna1",
        "antenna2",
        False,
    )

    updated_vis_mock.transpose.assert_called_once_with(0, 2, 1, 3)

    expected == updated_vis_mock.transpose.return_value


@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".apply.apply_antenna_gains_to_visibility"
)
def test_should_apply_gaintable_to_dataset_without_newaxis(
    apply_gaintable_mock,
):
    vis_mock = MagicMock(name="vis")
    pol_gain_mock = MagicMock(name="pol_gains")
    pol_gain_mock.shape = (1, 2, 4)
    gain_mock = pol_gain_mock.reshape.return_value
    gain_mock.__getitem__.return_value = gain_mock
    gain_mock.shape = (1, 2, 2, 2)

    _apply_gaintable_to_dataset_ufunc(
        vis_mock, pol_gain_mock, "antenna1", "antenna2", False
    )

    gain_mock.__getitem__.assert_called_once_with((np.newaxis, ...))


# Scalar (diagonal, equal X and Y) gain of each antenna, in antenna_xds order
ANTENNA_GAINS = np.array([1.0 + 1.0j, 2.0, 0.5j])
# antenna_xds indices of the baselines of the conftest MSv4 node
ANTENNA1, ANTENNA2 = np.array([0, 2, 0]), np.array([1, 1, 2])


def _scalar_gaintable(ms_xdt, timeslice, jones_type):
    gaintable = create_gaintable_from_msv4(ms_xdt, timeslice, jones_type)
    gain = gaintable.CALPARAM_GAIN
    # Diagonal Jones matrices: XX = YY = g, XY = YX = 0
    jones = np.zeros(gain.shape, dtype=gain.dtype)
    jones[..., 0] = ANTENNA_GAINS[np.newaxis, :, np.newaxis]
    jones[..., 3] = ANTENNA_GAINS[np.newaxis, :, np.newaxis]
    return gaintable.assign(
        CALPARAM_GAIN=gain.copy(data=da.from_array(jones, chunks=gain.chunks))
    )


@pytest.mark.parametrize("jones_type", ["B", "T"])
@pytest.mark.parametrize("inverse", [False, True])
def test_should_apply_gaintable_to_msv4(make_ms_xdt, jones_type, inverse):
    ms_xdt = make_ms_xdt()
    gaintable = _scalar_gaintable(ms_xdt, 6.0, jones_type)
    assert len(gaintable.soln_interval_slices) == 2

    applied = apply_gaintable_to_dataset(ms_xdt, gaintable, inverse=inverse)

    # V' = g1 V g2^H, using the antenna order of each baseline, including
    # the one with antenna1 > antenna2
    baseline_gain = ANTENNA_GAINS[ANTENNA1] * np.conj(ANTENNA_GAINS[ANTENNA2])
    if inverse:
        baseline_gain = 1 / baseline_gain
    expected = (
        ms_xdt.ds.VISIBILITY.values
        * baseline_gain[np.newaxis, :, np.newaxis, np.newaxis]
    )

    assert isinstance(applied.ds.VISIBILITY.data, da.Array)
    assert applied.ds.VISIBILITY.chunks[2] == ms_xdt.ds.VISIBILITY.chunks[2]
    np.testing.assert_allclose(
        applied.ds.VISIBILITY.values, expected, rtol=1e-6
    )
    assert applied.ds.FLAG.data is ms_xdt.ds.FLAG.data


def test_should_raise_applying_gaintable_with_other_antenna_order(
    make_ms_xdt,
):
    ms_xdt = make_ms_xdt()
    gaintable = _scalar_gaintable(ms_xdt, "full", "B")

    with pytest.raises(ValueError, match="antennas differ"):
        apply_gaintable_to_dataset(
            ms_xdt, gaintable.isel(antenna_name=[1, 0, 2])
        )


def test_should_raise_applying_gaintable_with_other_polarizations(
    make_ms_xdt,
):
    ms_xdt = make_ms_xdt()
    gaintable = _scalar_gaintable(ms_xdt, "full", "B")

    with pytest.raises(ValueError, match="polarizations"):
        apply_gaintable_to_dataset(
            ms_xdt,
            gaintable.assign_coords(polarization=["RR", "RL", "LR", "LL"]),
        )
