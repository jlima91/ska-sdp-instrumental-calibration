import h5py
import numpy as np

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_to_h5parm,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable


def test_export_gaintable_to_h5parm(generate_vis, tmp_path):
    _, gaintable = generate_vis

    gaintable = gaintable.copy(deep=True)
    # Forcefully set one of the crosspol weights to zero,
    # to check if gains are also zero
    new_weight = gaintable.weight
    new_weight[:, :, :, 0, 1] = 0.0
    gaintable = gaintable.assign(weight=new_weight)

    expected_masked_gains = gaintable.gain.data.copy()
    expected_masked_gains[:, :, :, 0, 1] = np.nan

    filename = str(tmp_path / "gaintable.h5parm")

    export_to_h5parm.export_gaintable_to_h5parm(gaintable, filename)

    with h5py.File(filename, "r") as h5f:
        solution = h5f["sol000"]
        amplitude = solution["amplitude000"]
        phase = solution["phase000"]

        gain = np.reshape(
            amplitude["val"][...] * np.exp(phase["val"][...] * 1j),
            gaintable.gain.shape,
        )

        # gaintable is complex64, while the data stored in h5parm is complex128
        # thus using lower tolerance for comparision
        np.testing.assert_allclose(expected_masked_gains, gain, atol=1e-7)

        weight_amp = amplitude["weight"][...]
        weight_phase = phase["weight"][...]

        # Assert that both amp and phase weights are equal
        np.testing.assert_allclose(weight_amp, weight_phase)

        # Assert that all weights equal to previous weights
        new_weight = np.reshape(gaintable.weight.data, weight_amp.shape)
        np.testing.assert_allclose(weight_amp, new_weight)

        # TODO: Add comparision for polarisation values.
        # Gaintable has "receptor1" and "receptor2", while h5parm has "pol"
        np.testing.assert_allclose(amplitude["time"][...], gaintable.time.data)
        np.testing.assert_allclose(
            amplitude["freq"][...], gaintable.frequency.data
        )
        assert np.all(
            amplitude["ant"][...].astype(str)
            == gaintable.configuration.names.data.astype(str)
        )


def test_export_clock_to_h5parm(generate_vis, tmp_path):
    _, gaintable = generate_vis
    rng = np.random.default_rng(42)
    shape = (2, gaintable.antenna.size, 2)
    delaytable = DelayTable.constructor(
        delay=rng.normal(scale=1e-9, size=shape),
        offset=rng.uniform(-0.5, 0.5, size=shape),
        time=gaintable.time.data[:2],
        antenna=gaintable.antenna.data,
        pol=["XX", "YY"],
        configuration=gaintable.configuration,
    )

    filename = str(tmp_path / "clock.h5parm")

    export_to_h5parm.export_clock_to_h5parm(delaytable, filename)

    with h5py.File(filename, "r") as h5f:
        clock = h5f["sol000"]["clock000"]

        assert clock.attrs["TITLE"] == b"clock"
        assert clock["val"].attrs["AXES"] == b"time,ant,pol"
        assert clock["offset"].attrs["AXES"] == b"time,ant,pol"

        np.testing.assert_allclose(clock["val"][...], delaytable.delay.data)
        np.testing.assert_allclose(
            clock["offset"][...], delaytable.offset.data
        )

        np.testing.assert_allclose(clock["time"][...], delaytable.time.data)
        assert list(clock["pol"][...].astype(str)) == ["XX", "YY"]
        assert np.all(
            clock["ant"][...].astype(str)
            == gaintable.configuration.names.data.astype(str)
        )
