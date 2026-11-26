import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_msv4,
)
from ska_sdp_instrumental_calibration.data_managers.processing_set import (
    with_correlated_data,
)
from ska_sdp_instrumental_calibration.data_managers.schema.calibration_set import (  # noqa: E501
    GainCalibrationSetXds,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import (
    DelayTable,
    apply_delay_to_gaintable,
    calculate_delays_from_gain,
    calculate_gain_rot,
    coarse_delay,
    create_delaytable_from_vis,
    update_delay,
)


def test_should_calculate_coarse_delay():
    oversample = 8
    nstations = 20
    nchan = 4
    frequency = np.linspace(100e6, 150e6, nchan)
    gains_per_station = np.ones(nchan) + 1j * np.ones(nchan)
    expected = np.zeros(nstations)
    actual = coarse_delay(gains_per_station, frequency, oversample)

    assert np.allclose(expected, actual, atol=1e-11)


def test_calculate_gain_rotation():
    frequency = np.linspace(100e6, 200e6, 4).reshape(1, -1)
    f = np.linspace(4, 10, 4) + 1j * np.linspace(3, 9, 4)
    gains = f
    delay = np.ones(1)
    offset = np.ones(1)

    expected = [
        [
            3.99999993 + 3.00000009j,
            1.33012709 - 7.69615241j,
            -10.06217822 + 3.42820208j,
            10.00000017 + 8.99999981j,
        ]
    ]
    actual = calculate_gain_rot(gains, delay, offset, frequency, inverse=True)

    np.testing.assert_allclose(actual, expected)

    actual = calculate_gain_rot(
        expected, delay, offset, frequency, inverse=False
    )

    np.testing.assert_allclose(actual, [gains])


def test_calculate_and_apply_delay():

    coords = {
        "antenna_name": ["antenna1", "antenna2"],
        "frequency": np.linspace(1.0010e8, 1.0019e8, 4, dtype=np.float64),
        "polarization": ["XX", "XY", "YX", "YY"],
    }

    # gain_shape = [ntimes, nants, nfrequency, nrec, nrec]
    gains = xr.DataArray(
        np.array(
            [
                np.cos(np.pi / 3) + 1j * np.sin(np.pi / 3),
                np.cos(2 * np.pi / 3) + 1j * np.sin(2 * np.pi / 3),
            ]
            * 16,
            dtype=np.complex128,
        ).reshape(1, 2, 4, 4),
        dims=[
            "time",
            "antenna_name",
            "frequency",
            "polarization",
        ],
    )

    weights = xr.DataArray(
        np.ones(32, dtype=np.float64).reshape(1, 2, 4, 4),
        dims=[
            "time",
            "antenna_name",
            "frequency",
            "polarization",
        ],
    )

    gaintable = xr.Dataset(
        {
            "CALPARAM_GAIN": gains,
            "CALPARAM_WEIGHT": weights,
        },
        coords=coords,
    )

    delay = calculate_delays_from_gain(gaintable, 4)

    assert delay.antenna_name.data.tolist() == ["antenna1", "antenna2"]

    actual_gaintable = apply_delay_to_gaintable(gaintable, delay, inverse=True)

    # Delay correction is applied to XX/YY only; XY/YX are zero-filled.
    np.testing.assert_allclose(
        np.angle(actual_gaintable.CALPARAM_GAIN.data[..., 0], deg=True),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.angle(actual_gaintable.CALPARAM_GAIN.data[..., 3], deg=True),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        actual_gaintable.CALPARAM_GAIN.data[..., 1], 0.0j
    )
    np.testing.assert_allclose(
        actual_gaintable.CALPARAM_GAIN.data[..., 2], 0.0j
    )


def _gaintable_with_known_delays(
    delay, offset, frequency
) -> GainCalibrationSetXds:
    """
    Build a gaintable whose diagonal gains are pure delay rotations.

    delay, offset: (time, antenna_name, pol)
    """
    ntime, nant, _ = delay.shape
    phase = (
        2
        * np.pi
        * (
            offset[:, :, np.newaxis, :]
            + delay[:, :, np.newaxis, :]
            * frequency[np.newaxis, np.newaxis, :, np.newaxis]
        )
    )
    gain = np.zeros((ntime, nant, frequency.size, 4), dtype=np.complex128)
    gain[..., 0] = np.exp(1j * phase[..., 0])
    gain[..., 3] = np.exp(1j * phase[..., 1])

    time_dim = np.arange(ntime, dtype=np.float64)
    interval_dim = np.ones_like(time_dim)

    return GainCalibrationSetXds.constructor(
        gain=gain,
        weight=np.ones(gain.shape),
        residual=np.zeros((ntime, frequency.size, 4)),
        time=time_dim,
        interval=interval_dim,
        frequency=frequency,
        antenna_name=np.arange(nant),
        polarization=np.asarray(["XX", "XY", "YX", "YY"]),
        jones_type="B",
    )


def test_coarse_delay_should_operate_on_leading_dimensions():
    oversample = 8
    nchan = 16
    frequency = np.linspace(100e6, 150e6, nchan)
    delays = np.array([[0.0, 1e-7], [-2e-7, 3e-7]])
    gains = np.exp(2j * np.pi * delays[..., np.newaxis] * frequency)

    actual = coarse_delay(gains, frequency, oversample)

    assert actual.shape == (2, 2, 1)
    for idx in np.ndindex(delays.shape):
        np.testing.assert_allclose(
            actual[idx], coarse_delay(gains[idx], frequency, oversample)
        )


def test_update_delay_should_operate_on_leading_dimensions():
    nchan = 16
    frequency = np.linspace(100e6, 150e6, nchan)
    delays = np.array([[1e-8, 2e-8], [-3e-8, 4e-8]])
    offsets = np.array([[0.1, -0.2], [0.05, 0.0]])
    gains = np.exp(
        2j
        * np.pi
        * (offsets[..., np.newaxis] + delays[..., np.newaxis] * frequency)
    )
    weights = np.ones(gains.shape)
    zeros = np.zeros((2, 2, 1))

    delay, offset = update_delay(gains, weights, frequency, zeros, zeros)

    assert delay.shape == (2, 2, 1)
    assert offset.shape == (2, 2, 1)
    np.testing.assert_allclose(delay[..., 0], delays, rtol=1e-6)
    # Offset is only defined up to an integer number of cycles
    cycle_diff = offset[..., 0] - offsets
    np.testing.assert_allclose(cycle_diff, np.round(cycle_diff), atol=1e-6)


def test_calculate_delays_from_gain_should_recover_known_delays():
    ntime, nant = 2, 3
    frequency = np.linspace(100e6, 150e6, 32)
    rng = np.random.default_rng(42)
    expected_delay = rng.uniform(-5e-8, 5e-8, (ntime, nant, 2))
    expected_offset = rng.uniform(-0.2, 0.2, (ntime, nant, 2))

    gaintable = _gaintable_with_known_delays(
        expected_delay, expected_offset, frequency
    ).chunk(time=1, antenna_name=2, frequency=8)

    delaytable = calculate_delays_from_gain(gaintable, oversample=16)

    assert list(delaytable.pol.values) == ["XX", "YY"]
    assert delaytable.delay.dims == ("time", "antenna_name", "pol")
    # Time and antenna_name chunks of the gaintable are preserved
    assert delaytable.delay.chunksizes["time"] == (1, 1)
    assert delaytable.delay.chunksizes["antenna_name"] == (2, 1)
    assert delaytable.offset.chunksizes["time"] == (1, 1)
    assert delaytable.offset.chunksizes["antenna_name"] == (2, 1)

    np.testing.assert_allclose(
        delaytable.delay.values, expected_delay, rtol=1e-6
    )
    np.testing.assert_allclose(
        delaytable.offset.values, expected_offset, atol=1e-6
    )


def test_apply_delay_to_gaintable_should_preserve_chunks():
    ntime, nant = 2, 3
    frequency = np.linspace(100e6, 150e6, 32)
    rng = np.random.default_rng(7)
    delay = rng.uniform(-5e-8, 5e-8, (ntime, nant, 2))
    offset = rng.uniform(-0.2, 0.2, (ntime, nant, 2))

    gaintable = _gaintable_with_known_delays(delay, offset, frequency)
    # Non-zero off-diagonal terms, which should be zeroed
    gaintable["CALPARAM_GAIN"][..., 1] = 1.0 + 1.0j
    gaintable["CALPARAM_GAIN"][..., 2] = 1.0 - 1.0j
    gaintable = gaintable.chunk(time=1, antenna_name=2, frequency=8)

    delaytable = DelayTable.constructor(
        delay=delay,
        offset=offset,
        time=gaintable.time.values,
        antenna_name=gaintable.antenna_name.values,
        pol=["XX", "YY"],
    )

    actual = apply_delay_to_gaintable(gaintable, delaytable, inverse=True)

    assert actual.CALPARAM_GAIN.dims == gaintable.CALPARAM_GAIN.dims
    assert (
        actual.CALPARAM_GAIN.chunksizes == gaintable.CALPARAM_GAIN.chunksizes
    )

    gain = actual.CALPARAM_GAIN.values
    np.testing.assert_allclose(gain[..., 0], 1.0 + 0.0j, atol=1e-6)
    np.testing.assert_allclose(gain[..., 3], 1.0 + 0.0j, atol=1e-6)
    np.testing.assert_allclose(gain[..., 1], 0.0j)
    np.testing.assert_allclose(gain[..., 2], 0.0j)

    # Applying the delays (non-inverse) restores the original diagonal
    restored = apply_delay_to_gaintable(actual, delaytable, inverse=False)
    np.testing.assert_allclose(
        restored.CALPARAM_GAIN.values[..., 0],
        gaintable.CALPARAM_GAIN.values[..., 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        restored.CALPARAM_GAIN.values[..., 3],
        gaintable.CALPARAM_GAIN.values[..., 3],
        atol=1e-6,
    )


# Delays of the 3 antennas of the conftest MSv4 node, small enough for the
# phase to not wrap between its 6 channels of 1 MHz
_ANTENNA_DELAYS = np.array([3e-8, -2e-8, 5e-8])
# The delay fit, in absolute frequency, loses precision with float32
# weights (as stored in MSs): ~1e-4 relative error for exact data. A wrong
# antenna or sign would be off by more than 100%.
_DELAY_RTOL = 1e-3


def _node_with_antenna_delays(make_ms_xdt):
    """
    MSv4 node whose visibilities are V(a1, a2) = g(a1) conj(g(a2)), with
    gains g(a) = exp(2 pi i delay(a) frequency) on XX and YY.
    """
    ms_xdt = make_ms_xdt()
    ds = ms_xdt.ds
    antenna1, antenna2 = (
        np.array([0, 2, 0]),
        np.array([1, 1, 2]),
    )  # conftest baselines (a0, a1), (a2, a1), (a0, a2)
    frequency = ds.frequency.values
    gain = np.exp(2j * np.pi * _ANTENNA_DELAYS[:, None] * frequency)
    baseline_vis = gain[antenna1] * np.conj(gain[antenna2])
    visibility = np.zeros(ds.VISIBILITY.shape, dtype=ds.VISIBILITY.dtype)
    visibility[..., 0] = baseline_vis[None]
    visibility[..., 3] = baseline_vis[None]
    return with_correlated_data(
        ms_xdt,
        ds.VISIBILITY.copy(
            data=da.from_array(visibility, chunks=ds.VISIBILITY.chunks)
        ),
    )


@pytest.mark.parametrize("refant", [0, 1, 2])
def test_create_delaytable_from_vis(make_ms_xdt, refant):
    # refant 0: both baselines stored as (refant, antenna), conjugated
    # refant 1: both stored as (antenna, refant)
    # refant 2: one of each
    vis = _node_with_antenna_delays(make_ms_xdt)
    gaintable = create_gaintable_from_msv4(vis, 6.0, "B")
    assert len(gaintable.soln_interval_slices) == 2

    delaytable = create_delaytable_from_vis(vis, gaintable, refant, 1)

    assert isinstance(delaytable, xr.Dataset)
    assert delaytable.delay.shape == (2, 3, 2)
    np.testing.assert_array_equal(
        delaytable.antenna_name.data, gaintable.antenna_name.data
    )
    np.testing.assert_array_equal(delaytable.time, gaintable.time)
    # Delay of each antenna relative to the reference antenna, on XX and YY
    expected = _ANTENNA_DELAYS - _ANTENNA_DELAYS[refant]
    np.testing.assert_allclose(
        delaytable.delay.values,
        np.broadcast_to(expected[None, :, None], (2, 3, 2)),
        rtol=_DELAY_RTOL,
    )


def test_create_delaytable_from_vis_without_refant_baseline(make_ms_xdt):
    vis = _node_with_antenna_delays(make_ms_xdt)
    # Drop (a0, a2): antenna 0 has no baseline with refant 2
    without_baseline = vis.copy(deep=False)
    without_baseline.ds = vis.to_dataset().isel(baseline_id=[0, 1])
    gaintable = create_gaintable_from_msv4(without_baseline, "full", "B")

    delaytable = create_delaytable_from_vis(without_baseline, gaintable, 2, 1)

    # Antenna 0 gets zero visibilities and weights, as missing (zero
    # filled) baselines did in the MSv2 Visibility. The weighted fit is
    # then undefined, giving NaN, as before migration.
    with np.errstate(invalid="ignore"):
        delay, offset = dask.compute(
            delaytable.delay.data, delaytable.offset.data
        )
    assert np.isnan(delay[0, 0]).all()
    assert np.isnan(offset[0, 0]).all()
    np.testing.assert_allclose(
        delay[0, 1],
        _ANTENNA_DELAYS[1] - _ANTENNA_DELAYS[2],
        rtol=_DELAY_RTOL,
    )
