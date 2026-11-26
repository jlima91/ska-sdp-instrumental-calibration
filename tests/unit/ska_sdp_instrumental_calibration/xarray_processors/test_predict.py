import dask.array as da
import numpy as np
import pytest
from mock import MagicMock
from ska_sdp_datamodels.sky_model import SkyComponent

from ska_sdp_instrumental_calibration.data_managers.processing_set import (
    get_phase_centre,
)
from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    LocalSkyModel,
)
from ska_sdp_instrumental_calibration.xarray_processors.predict import (
    _predict_vis_ufunc,
    predict_vis,
)


def _object_scalar(obj):
    wrapped = np.empty((), dtype=object)
    wrapped[()] = obj
    return wrapped


def test_should_create_vis_for_provided_sky_model():
    uvw_mock = MagicMock(name="uvw")
    uvw_mock.reshape.return_value = uvw_mock
    uvw_mock.shape = (1, 2, 3, 4, 5)
    local_sky_model = MagicMock(name="local_sky_model")
    created_vis = MagicMock(name="vis")
    local_sky_model.create_vis.return_value = created_vis

    expected = _predict_vis_ufunc(
        uvw_mock,
        _object_scalar(local_sky_model),
        "frequency",
        _object_scalar("beams_factory"),
        "station_rm",
        "polarisation",
        "antenna1",
        "antenna2",
        "phasecentre",
        "output_dtype",
    )

    uvw_mock.reshape.assert_called_once_with(1, 3, 4, 5)

    local_sky_model.create_vis.assert_called_once_with(
        uvw_mock,
        "frequency",
        "polarisation",
        "phasecentre",
        "antenna1",
        "antenna2",
        "beams_factory",
        "station_rm",
        "output_dtype",
    )

    created_vis.transpose.assert_called_once_with(0, 2, 1, 3)
    assert expected == created_vis.transpose.return_value


SOLN_TIME = np.array([5.1e9, 5.2e9])
SOLN_INTERVAL_SLICES = [slice(0, 2), slice(2, 4)]


class _StubGlobalSkyModel:
    """Global sky model returning one off-axis component at every time."""

    def __init__(self):
        self.calls = []

    def get_local_sky_model(self, solution_time, array_location):
        self.calls.append((solution_time, array_location))
        component = SkyComponent(
            component_id="c0",
            source_id="",
            ra_deg=198.2,
            dec_deg=-22.0,
            i_pol_jy=3.0,
            ref_freq_hz=1.2e8,
            epoch=0.0,
            log_spec_idx=True,
        )
        return LocalSkyModel([component], solution_time)


def _expected_prediction(ms_xdt, gsm_calls, station_rm=None):
    """Prediction computed directly with LocalSkyModel.create_vis."""
    ds = ms_xdt.ds
    # Baseline antenna indices in antenna_xds order: a0=0, a1=1, a2=2
    antenna1, antenna2 = np.array([0, 2, 0]), np.array([1, 1, 2])
    # The prediction uses the negated MSv4 UVW
    uvw = -ds.UVW.values
    return np.concatenate(
        [
            LocalSkyModel(
                [
                    _StubGlobalSkyModel()
                    .get_local_sky_model(time, None)
                    .components[0]
                ],
                time,
            ).create_vis(
                uvw[slc],
                ds.frequency.values,
                ds.polarization.values,
                get_phase_centre(ms_xdt),
                antenna1,
                antenna2,
                None,
                station_rm,
                np.complex64,
            )
            for (time, _), slc in zip(gsm_calls, SOLN_INTERVAL_SLICES)
        ]
    )


def test_should_predict_msv4_visibilities_lazily(make_ms_xdt):
    ms_xdt = make_ms_xdt()
    gsm = _StubGlobalSkyModel()

    modelvis = predict_vis(ms_xdt, gsm, SOLN_TIME, SOLN_INTERVAL_SLICES)

    predicted = modelvis.ds.VISIBILITY
    # Lazy, with the frequency chunks of the input
    assert isinstance(predicted.data, da.Array)
    assert predicted.chunks[2] == ms_xdt.ds.VISIBILITY.chunks[2]
    assert predicted.dims == ms_xdt.ds.VISIBILITY.dims
    assert predicted.dtype == np.complex64

    # One local sky model per solution interval, at the solution time and
    # the location of the first antenna
    assert [time for time, _ in gsm.calls] == list(SOLN_TIME)
    for _, location in gsm.calls:
        np.testing.assert_allclose(
            [c.to_value("m") for c in location.geocentric],
            ms_xdt["antenna_xds"].ds.ANTENNA_POSITION.values[0],
        )

    np.testing.assert_allclose(
        predicted.values, _expected_prediction(ms_xdt, gsm.calls), rtol=1e-6
    )


def test_should_keep_other_msv4_variables_and_children(make_ms_xdt):
    ms_xdt = make_ms_xdt()

    modelvis = predict_vis(
        ms_xdt, _StubGlobalSkyModel(), SOLN_TIME, SOLN_INTERVAL_SLICES
    )

    assert modelvis.ds.FLAG.data is ms_xdt.ds.FLAG.data
    assert modelvis.ds.UVW.data is ms_xdt.ds.UVW.data
    assert set(modelvis.children) == set(ms_xdt.children)
    # The input is left untouched
    assert not np.array_equal(
        modelvis.ds.VISIBILITY.values, ms_xdt.ds.VISIBILITY.values
    )


def test_should_predict_msv4_with_station_rotation_measures(make_ms_xdt):
    ms_xdt = make_ms_xdt()
    gsm = _StubGlobalSkyModel()
    station_rm = np.array([0.5, -1.0, 2.0])

    modelvis = predict_vis(
        ms_xdt,
        gsm,
        SOLN_TIME,
        SOLN_INTERVAL_SLICES,
        station_rm=da.from_array(station_rm),
    )

    np.testing.assert_allclose(
        modelvis.ds.VISIBILITY.values,
        _expected_prediction(ms_xdt, gsm.calls, station_rm),
        rtol=1e-6,
    )


def test_should_raise_predicting_non_linear_polarizations(make_ms_xdt):
    ms_xdt = make_ms_xdt(polarization=("RR", "RL", "LR", "LL"))

    with pytest.raises(ValueError, match="requires polarizations"):
        predict_vis(
            ms_xdt, _StubGlobalSkyModel(), SOLN_TIME, SOLN_INTERVAL_SLICES
        )


def test_should_assert_on_msv4_solution_length(make_ms_xdt):
    with pytest.raises(AssertionError, match="do not match"):
        predict_vis(
            make_ms_xdt(),
            _StubGlobalSkyModel(),
            SOLN_TIME,
            SOLN_INTERVAL_SLICES[:1],
        )
