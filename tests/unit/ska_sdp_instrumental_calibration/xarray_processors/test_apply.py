import numpy as np
from mock import MagicMock, call, patch

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
    gain_mock = MagicMock(name="gains")
    updated_vis_mock = MagicMock(name="applied_vis")
    apply_gaintable_mock.return_value = updated_vis_mock

    gain_mock.__getitem__.return_value = gain_mock
    gain_mock.shape = (1, 2, 3)

    expected = _apply_gaintable_to_dataset_ufunc(
        vis_mock, gain_mock, "antenna1", "antenna2", False
    )

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
    gain_mock = MagicMock(name="gains")
    gain_mock.__getitem__.return_value = gain_mock
    gain_mock.shape = (1, 2, 3, 4)

    _apply_gaintable_to_dataset_ufunc(
        vis_mock, gain_mock, "antenna1", "antenna2", False
    )

    gain_mock.__getitem__.assert_called_once_with((np.newaxis, ...))


@patch("ska_sdp_instrumental_calibration.xarray_processors.apply.xr.concat")
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".apply.xr.apply_ufunc"
)
def test_should_apply_gaintable_to_visibility(apply_ufunc_mock, concat_mock):
    vis_mock = MagicMock(name="vis")
    vis_mock.chunksizes = {"frequency": 1, "time": 1}

    gaintable = MagicMock(name="gaintable")
    gains = MagicMock(name="gains")
    gains.chunk.return_value = gains
    gaintable.gain = gains

    gaintable.jones_type = "B"
    gaintable.soln_interval_slices = [1, 2]
    transpose_mock = MagicMock(
        name="Transpose", side_effect=["TRANSPOSE_1", "TRANSPOSE_2"]
    )

    concated_ds_mock = MagicMock(name="Concated DS")
    concat_mock.return_value = concated_ds_mock
    concated_ds_mock.assign_attrs.return_value = concated_ds_mock

    apply_ufunc_mock.return_value.transpose = transpose_mock

    expected = apply_gaintable_to_dataset(vis_mock, gaintable, False)

    apply_ufunc_mock.assert_has_calls(
        [
            call(
                _apply_gaintable_to_dataset_ufunc,
                vis_mock.vis.isel.return_value,
                gains.isel.return_value,
                input_core_dims=[
                    ["baselineid", "polarisation"],
                    ["antenna", "receptor1", "receptor2"],
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
                kwargs={
                    "antenna1": vis_mock.antenna1,
                    "antenna2": vis_mock.antenna2,
                    "inverse": False,
                },
            ),
            call(
                _apply_gaintable_to_dataset_ufunc,
                vis_mock.vis.isel.return_value,
                gains.isel.return_value,
                input_core_dims=[
                    ["baselineid", "polarisation"],
                    ["antenna", "receptor1", "receptor2"],
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
                kwargs={
                    "antenna1": vis_mock.antenna1,
                    "antenna2": vis_mock.antenna2,
                    "inverse": False,
                },
            ),
        ]
    )

    gains.chunk.assert_called_once_with({"frequency": 1})

    concat_mock.assert_called_once_with(
        ["TRANSPOSE_1", "TRANSPOSE_2"], dim="time"
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


@patch("ska_sdp_instrumental_calibration.xarray_processors.apply.xr.concat")
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".apply.xr.apply_ufunc"
)
def test_should_apply_gaintable_to_visibility_for_non_B_type(
    apply_ufunc_mock, concat_mock
):
    vis_mock = MagicMock(name="vis")
    vis_mock.chunksizes = {"frequency": 1, "time": 1}

    gaintable = MagicMock(name="gaintable")
    gains = MagicMock(name="gains")
    gains.chunk.return_value = gains
    gaintable.gain = gains

    gaintable.jones_type = "G"
    gaintable.soln_interval_slices = [1]
    gains.frequency.size = 1

    apply_gaintable_to_dataset(vis_mock, gaintable, False)
    gains.isel.assert_called_once_with(frequency=0, drop=True)
