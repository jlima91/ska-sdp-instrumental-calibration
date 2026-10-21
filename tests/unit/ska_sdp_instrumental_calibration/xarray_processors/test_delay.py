import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors._utils import (
    simplify_baselines_dim,
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
        "antenna": ["antenna1", "antenna2"],
        "frequency": np.linspace(1.0010e8, 1.0019e8, 4, dtype=np.float64),
        "receptor1": ["X", "Y"],
        "receptor2": ["X", "Y"],
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
        ).reshape(1, 2, 4, 2, 2),
        dims=["time", "antenna", "frequency", "receptor1", "receptor2"],
    )

    weights = xr.DataArray(
        np.ones(32, dtype=np.float64).reshape(1, 2, 4, 2, 2),
        dims=["time", "antenna", "frequency", "receptor1", "receptor2"],
    )

    gaintable = xr.Dataset(
        {
            "gain": gains,
            "weight": weights,
        },
        coords=coords,
        attrs={"configuration": "Antenna Configuration"},
    )

    delay = calculate_delays_from_gain(gaintable, 4)

    actual_gaintable = apply_delay_to_gaintable(gaintable, delay, inverse=True)

    # Delay correction is applied to XX/YY only; XY/YX are zero-filled.
    np.testing.assert_allclose(
        np.angle(actual_gaintable.gain.data[..., 0, 0], deg=True),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.angle(actual_gaintable.gain.data[..., 1, 1], deg=True),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(actual_gaintable.gain.data[..., 0, 1], 0.0j)
    np.testing.assert_allclose(actual_gaintable.gain.data[..., 1, 0], 0.0j)


def _gaintable_with_known_delays(delay, offset, frequency):
    """
    Build a gaintable whose diagonal gains are pure delay rotations.

    delay, offset: (time, antenna, pol)
    """
    ntime, nant, _ = delay.shape
    phase = (
        2
        * np.pi
        * (
            offset[:, :, np.newaxis, :]
            + delay[:, :, np.newaxis, :] * frequency[:, np.newaxis]
        )
    )
    gain = np.zeros((ntime, nant, frequency.size, 2, 2), dtype=np.complex128)
    gain[..., 0, 0] = np.exp(1j * phase[..., 0])
    gain[..., 1, 1] = np.exp(1j * phase[..., 1])

    dims = ["time", "antenna", "frequency", "receptor1", "receptor2"]
    return xr.Dataset(
        {
            "gain": xr.DataArray(gain, dims=dims),
            "weight": xr.DataArray(np.ones(gain.shape), dims=dims),
        },
        coords={
            "time": np.arange(ntime, dtype=np.float64),
            "antenna": np.arange(nant),
            "frequency": frequency,
            "receptor1": ["X", "Y"],
            "receptor2": ["X", "Y"],
        },
        attrs={"configuration": None},
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
    ).chunk(time=1, antenna=2, frequency=8)

    delaytable = calculate_delays_from_gain(gaintable, oversample=16)

    assert list(delaytable.pol.values) == ["XX", "YY"]
    assert delaytable.delay.dims == ("time", "antenna", "pol")
    # Time and antenna chunks of the gaintable are preserved
    assert delaytable.delay.chunksizes["time"] == (1, 1)
    assert delaytable.delay.chunksizes["antenna"] == (2, 1)
    assert delaytable.offset.chunksizes["time"] == (1, 1)
    assert delaytable.offset.chunksizes["antenna"] == (2, 1)

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
    gaintable["gain"][..., 0, 1] = 1.0 + 1.0j
    gaintable["gain"][..., 1, 0] = 1.0 - 1.0j
    gaintable = gaintable.chunk(time=1, antenna=2, frequency=8)

    delaytable = DelayTable.constructor(
        delay=delay,
        offset=offset,
        time=gaintable.time.values,
        antenna=gaintable.antenna.values,
        pol=["XX", "YY"],
    )

    actual = apply_delay_to_gaintable(gaintable, delaytable, inverse=True)

    assert actual.gain.dims == gaintable.gain.dims
    assert actual.gain.chunksizes == gaintable.gain.chunksizes

    gain = actual.gain.values
    np.testing.assert_allclose(gain[..., 0, 0], 1.0 + 0.0j, atol=1e-6)
    np.testing.assert_allclose(gain[..., 1, 1], 1.0 + 0.0j, atol=1e-6)
    np.testing.assert_allclose(gain[..., 0, 1], 0.0j)
    np.testing.assert_allclose(gain[..., 1, 0], 0.0j)

    # Applying the delays (non-inverse) restores the original diagonal
    restored = apply_delay_to_gaintable(actual, delaytable, inverse=False)
    np.testing.assert_allclose(
        restored.gain.values[..., 0, 0],
        gaintable.gain.values[..., 0, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        restored.gain.values[..., 1, 1],
        gaintable.gain.values[..., 1, 1],
        atol=1e-6,
    )


def test_create_delaytable_from_vis(generate_vis):

    refant = 0
    oversample = 4

    vis, gaintable = generate_vis
    vis = simplify_baselines_dim(vis)

    actual_delaytable = create_delaytable_from_vis(
        vis, gaintable, refant, oversample
    )

    assert isinstance(actual_delaytable, xr.Dataset)
    assert actual_delaytable.delay.shape == (
        gaintable.dims["time"],
        gaintable.dims["antenna"],
        2,
    )
    assert actual_delaytable.offset.shape == (
        gaintable.dims["time"],
        gaintable.dims["antenna"],
        2,
    )

    np.testing.assert_allclose(actual_delaytable.delay.values, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        actual_delaytable.offset.values, 0.0, atol=1e-10
    )
