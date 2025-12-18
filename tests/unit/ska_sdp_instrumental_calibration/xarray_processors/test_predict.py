import dask.array as da
import pytest
from mock import ANY, MagicMock, call, patch

from ska_sdp_instrumental_calibration.xarray_processors.predict import (
    _predict_vis_ufunc,
    predict_vis,
    with_chunks,
)


def test_should_create_vis_for_provided_sky_model():
    uvw_mock = MagicMock(name="uvw")
    uvw_mock.reshape.return_value = uvw_mock
    uvw_mock.shape = (1, 2, 3, 4, 5)
    local_sky_model = MagicMock(name="local_sky_model")
    created_vis = MagicMock(name="vis")
    local_sky_model.create_vis.return_value = created_vis

    expected = _predict_vis_ufunc(
        uvw_mock,
        "frequency",
        "station_rm",
        "polarisation",
        "antenna1",
        "antenna2",
        "phasecentre",
        local_sky_model,
        "beams_factory",
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


def test_should_assert_on_solution_length():
    vis_mock = MagicMock(name="vis")
    vis_mock.chunksizes = {"frequency": 1, "time": 1}

    gsm_mock = MagicMock(name="gsm")
    soln_time = [10, 20]
    soln_interval_time = [1, 2, 3]

    with pytest.raises(
        AssertionError,
        match="lengths of soln_interval_slices and soln_time do not match",
    ):
        predict_vis(
            vis_mock,
            gsm_mock,
            soln_time,
            soln_interval_time,
        )


@patch("ska_sdp_instrumental_calibration.xarray_processors.predict.xr.concat")
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".predict.xr.apply_ufunc"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".predict.xr.DataArray"
)
def test_should_predict_visibilities(
    data_array_mock, apply_ufunc_mock, concat_mock
):
    vis_mock = MagicMock(name="vis")
    vis_mock.chunksizes = {"frequency": 1, "time": 1}

    gsm_mock = MagicMock(name="gsm")
    soln_time = [10, 20]
    soln_interval_time = [1, 2]

    beams_factory_mock = MagicMock(name="beam_factory")
    station_rm_mock = MagicMock(name="station_rm", spec=da.Array)

    frequency_mock = MagicMock(name="frequency")
    frequency_mock.pipe.return_value = frequency_mock

    transpose_mock = MagicMock(
        name="Transpose", side_effect=["TRANSPOSE_1", "TRANSPOSE_2"]
    )

    station_rm_da_mock = MagicMock(name="station_rm_da")
    station_rm_da_mock.chunk.return_value = station_rm_da_mock
    station_rm_da_mock.dims = ["dims"]

    concated_ds_mock = MagicMock(name="concatenated_ds_mock")
    concated_ds_mock.assign_attrs.return_value = concated_ds_mock
    concat_mock.return_value = concated_ds_mock

    data_array_mock.side_effect = [frequency_mock, station_rm_da_mock]
    apply_ufunc_mock.return_value.transpose = transpose_mock

    expected = predict_vis(
        vis_mock,
        gsm_mock,
        soln_time,
        soln_interval_time,
        beams_factory_mock,
        station_rm_mock,
    )

    data_array_mock.assert_has_calls(
        [
            call(vis_mock.frequency, name="frequency_xdr"),
            call(station_rm_mock, coords={"id": ANY}),
        ]
    )

    frequency_mock.pipe.assert_called_once_with(
        with_chunks, vis_mock.chunksizes
    )

    station_rm_da_mock.chunk.assert_called_once_with(-1)

    gsm_mock.get_local_sky_model.assert_has_calls(
        [
            call(10, vis_mock.configuration.location),
            call(20, vis_mock.configuration.location),
        ]
    )

    vis_mock.uvw.isel.assert_has_calls([call(time=1), call(time=2)])

    apply_ufunc_mock.assert_has_calls(
        [
            call(
                _predict_vis_ufunc,
                vis_mock.uvw.isel.return_value,
                frequency_mock,
                station_rm_da_mock,
                input_core_dims=[
                    ["baselineid", "spatial"],
                    [],
                    ["dims"],
                ],
                output_core_dims=[
                    ["baselineid", "polarisation"],
                ],
                dask="parallelized",
                output_dtypes=[vis_mock.vis.dtype],
                dask_gufunc_kwargs=dict(
                    output_sizes={
                        "baselineid": vis_mock.baselineid.size,
                        "polarisation": vis_mock.polarisation.size,
                    }
                ),
                kwargs=dict(
                    polarisation=vis_mock.polarisation,
                    antenna1=vis_mock.antenna1,
                    antenna2=vis_mock.antenna2,
                    phasecentre=vis_mock.phasecentre,
                    beams_factory=beams_factory_mock,
                    output_dtype=vis_mock.vis.dtype,
                    local_sky_model=gsm_mock.get_local_sky_model.return_value,
                ),
            ),
            call(
                _predict_vis_ufunc,
                vis_mock.uvw.isel.return_value,
                frequency_mock,
                station_rm_da_mock,
                input_core_dims=[
                    ["baselineid", "spatial"],
                    [],
                    ["dims"],
                ],
                output_core_dims=[
                    ["baselineid", "polarisation"],
                ],
                dask="parallelized",
                output_dtypes=[vis_mock.vis.dtype],
                dask_gufunc_kwargs=dict(
                    output_sizes={
                        "baselineid": vis_mock.baselineid.size,
                        "polarisation": vis_mock.polarisation.size,
                    }
                ),
                kwargs=dict(
                    polarisation=vis_mock.polarisation,
                    antenna1=vis_mock.antenna1,
                    antenna2=vis_mock.antenna2,
                    phasecentre=vis_mock.phasecentre,
                    beams_factory=beams_factory_mock,
                    output_dtype=vis_mock.vis.dtype,
                    local_sky_model=gsm_mock.get_local_sky_model.return_value,
                ),
            ),
        ]
    )

    transpose_mock.assert_has_calls(
        [
            call("time", "baselineid", "frequency", "polarisation"),
            call("time", "baselineid", "frequency", "polarisation"),
        ]
    )

    concated_ds_mock.assign_attrs.assert_called_once_with(vis_mock.vis.attrs)
    vis_mock.assign.assert_called_once_with({"vis": concated_ds_mock})
    assert expected == vis_mock.assign.return_value


@patch("ska_sdp_instrumental_calibration.xarray_processors.predict.xr.concat")
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".predict.xr.apply_ufunc"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".predict.xr.DataArray"
)
def test_should_predict_visibilities_non_da_type_station_rm(
    data_array_mock, apply_ufunc_mock, concat_mock
):
    vis_mock = MagicMock(name="vis")
    vis_mock.chunksizes = {"frequency": 1, "time": 1}

    gsm_mock = MagicMock(name="gsm")
    soln_time = [10]
    soln_interval_time = [1]

    beams_factory_mock = MagicMock(name="beam_factory")
    station_rm_mock = MagicMock(name="station_rm")
    station_rm_mock.chunk.return_value = station_rm_mock
    station_rm_mock.dims = ["dims"]

    frequency_mock = MagicMock(name="frequency")
    frequency_mock.pipe.return_value = frequency_mock

    transpose_mock = MagicMock(
        name="Transpose", side_effect=["TRANSPOSE_1", "TRANSPOSE_2"]
    )

    concated_ds_mock = MagicMock(name="concatenated_ds_mock")
    concated_ds_mock.assign_attrs.return_value = concated_ds_mock
    concat_mock.return_value = concated_ds_mock

    data_array_mock.return_value = frequency_mock
    apply_ufunc_mock.return_value.transpose = transpose_mock

    predict_vis(
        vis_mock,
        gsm_mock,
        soln_time,
        soln_interval_time,
        beams_factory_mock,
        station_rm_mock,
    )

    data_array_mock.assert_called_once_with(
        vis_mock.frequency, name="frequency_xdr"
    )

    station_rm_mock.chunk.assert_called_once_with(-1)

    apply_ufunc_mock.assert_called_once_with(
        _predict_vis_ufunc,
        vis_mock.uvw.isel.return_value,
        frequency_mock,
        station_rm_mock,
        input_core_dims=[
            ["baselineid", "spatial"],
            [],
            ["dims"],
        ],
        output_core_dims=[
            ["baselineid", "polarisation"],
        ],
        dask="parallelized",
        output_dtypes=[vis_mock.vis.dtype],
        dask_gufunc_kwargs=dict(
            output_sizes={
                "baselineid": vis_mock.baselineid.size,
                "polarisation": vis_mock.polarisation.size,
            }
        ),
        kwargs=dict(
            polarisation=vis_mock.polarisation,
            antenna1=vis_mock.antenna1,
            antenna2=vis_mock.antenna2,
            phasecentre=vis_mock.phasecentre,
            beams_factory=beams_factory_mock,
            output_dtype=vis_mock.vis.dtype,
            local_sky_model=gsm_mock.get_local_sky_model.return_value,
        ),
    )


@patch("ska_sdp_instrumental_calibration.xarray_processors.predict.xr.concat")
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".predict.xr.apply_ufunc"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".predict.xr.DataArray"
)
def test_should_predict_visibilities_none_station_rm(
    data_array_mock, apply_ufunc_mock, concat_mock
):
    vis_mock = MagicMock(name="vis")
    vis_mock.chunksizes = {"frequency": 1, "time": 1}

    gsm_mock = MagicMock(name="gsm")
    soln_time = [10]
    soln_interval_time = [1]

    beams_factory_mock = MagicMock(name="beam_factory")

    frequency_mock = MagicMock(name="frequency")
    frequency_mock.pipe.return_value = frequency_mock

    transpose_mock = MagicMock(
        name="Transpose", side_effect=["TRANSPOSE_1", "TRANSPOSE_2"]
    )

    concated_ds_mock = MagicMock(name="concatenated_ds_mock")
    concated_ds_mock.assign_attrs.return_value = concated_ds_mock
    concat_mock.return_value = concated_ds_mock

    data_array_mock.return_value = frequency_mock
    apply_ufunc_mock.return_value.transpose = transpose_mock

    predict_vis(
        vis_mock,
        gsm_mock,
        soln_time,
        soln_interval_time,
        beams_factory_mock,
    )

    data_array_mock.assert_called_once_with(
        vis_mock.frequency, name="frequency_xdr"
    )

    apply_ufunc_mock.assert_called_once_with(
        _predict_vis_ufunc,
        vis_mock.uvw.isel.return_value,
        frequency_mock,
        input_core_dims=[
            ["baselineid", "spatial"],
            [],
        ],
        output_core_dims=[
            ["baselineid", "polarisation"],
        ],
        dask="parallelized",
        output_dtypes=[vis_mock.vis.dtype],
        dask_gufunc_kwargs=dict(
            output_sizes={
                "baselineid": vis_mock.baselineid.size,
                "polarisation": vis_mock.polarisation.size,
            }
        ),
        kwargs=dict(
            polarisation=vis_mock.polarisation,
            station_rm=None,
            antenna1=vis_mock.antenna1,
            antenna2=vis_mock.antenna2,
            phasecentre=vis_mock.phasecentre,
            beams_factory=beams_factory_mock,
            output_dtype=vis_mock.vis.dtype,
            local_sky_model=gsm_mock.get_local_sky_model.return_value,
        ),
    )
