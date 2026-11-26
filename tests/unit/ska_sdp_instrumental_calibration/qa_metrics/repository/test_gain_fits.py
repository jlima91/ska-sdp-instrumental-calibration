import dask.array as da
import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.qa_metrics.repository.gain_fits import (
    export_gain_fits_to_zarr,
)


def _make_fit(seed):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        da.from_array(rng.random((2, 3, 4, 2, 2)), chunks=(1, 1, 4, 2, 2)),
        dims=["time", "antenna", "frequency", "receptor1", "receptor2"],
        coords={
            "time": [0.0, 1.0],
            "frequency": np.linspace(1e8, 2e8, 4),
            "receptor1": ["X", "Y"],
            "receptor2": ["X", "Y"],
        },
    )


def test_should_write_fits_and_attrs_to_zarr(tmp_path):
    fits = {"amp_fit": _make_fit(0), "phase_fit": _make_fit(1)}
    filename = str(tmp_path / "gain_flag.fits.zarr")
    attrs = {"soltype": "amp-phase", "normalize_gains": 1}

    export_gain_fits_to_zarr(fits, filename, attrs=attrs)

    actual = xr.open_zarr(filename).compute()

    assert set(actual.data_vars) == {"amp_fit", "phase_fit"}
    assert actual.attrs == attrs
    for name, fit in fits.items():
        xr.testing.assert_allclose(actual[name], fit.compute())


def test_should_overwrite_existing_store(tmp_path):
    filename = str(tmp_path / "gain_flag.fits.zarr")

    export_gain_fits_to_zarr({"amp_fit": _make_fit(0)}, filename)
    export_gain_fits_to_zarr({"phase_fit": _make_fit(1)}, filename)

    actual = xr.open_zarr(filename)

    assert set(actual.data_vars) == {"phase_fit"}
    assert actual.attrs == {}
