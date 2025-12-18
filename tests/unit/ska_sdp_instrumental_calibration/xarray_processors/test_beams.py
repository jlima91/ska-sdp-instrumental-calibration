import numpy as np
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.xarray_processors.beams import (
    _prediction_central_beams_ufunc,
    prediction_central_beams,
    with_chunks,
)


def test_should_predict_central_beams_for_numpy():
    beams_mock = MagicMock(name="beams")
    array_response_mock = MagicMock(name="array_response")

    beams_mock.array_response.return_value = array_response_mock
    beams_mock.get_beams_low.return_value = beams_mock

    expected = _prediction_central_beams_ufunc("frequency", 0.23, beams_mock)

    beams_mock.get_beams_low.assert_called_once_with("frequency", 0.23)
    beams_mock.array_response.assert_called_once_with(
        direction=beams_mock.beam_direction
    )

    array_response_mock.transpose.assert_called_once_with(1, 0, 2, 3)

    assert expected == array_response_mock.transpose.return_value


@patch("ska_sdp_instrumental_calibration.xarray_processors.beams.xr.concat")
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors"
    ".beams.xr.apply_ufunc"
)
@patch(
    "ska_sdp_instrumental_calibration.xarray_processors" ".beams.xr.DataArray"
)
def test_should_predict_central_beams(
    data_array_mock, apply_ufunc_mock, concat_mock
):
    gaintable_mock = MagicMock(name="gaintable")
    gaintable_mock.time.data = [1, 2]
    beams_factory = MagicMock(name="beams_factory")
    frequency_mock = MagicMock(name="frequency")
    frequency_mock.pipe.return_value = frequency_mock

    transpose_mock = MagicMock(
        name="Transpose", side_effect=["TRANSPOSE_1", "TRANSPOSE_2"]
    )

    apply_ufunc_mock.return_value.transpose = transpose_mock
    response_mock = MagicMock(name="concatenated_response")
    concat_mock.return_value = response_mock
    response_mock.assign_coords.return_value = response_mock
    response_mock.assign_attrs.return_value = response_mock

    data_array_mock.return_value = frequency_mock

    expected = prediction_central_beams(gaintable_mock, beams_factory)

    data_array_mock.assert_called_once_with(
        gaintable_mock.frequency, name="frequency_xdr"
    )
    frequency_mock.pipe.assert_called_once_with(
        with_chunks, gaintable_mock.chunksizes
    )

    apply_ufunc_mock.assert_has_calls(
        [
            call(
                _prediction_central_beams_ufunc,
                frequency_mock,
                input_core_dims=[[]],
                output_core_dims=[("antenna", "receptor1", "receptor2")],
                dask="parallelized",
                output_dtypes=[
                    np.complex128,
                ],
                join="outer",
                dataset_join="outer",
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "antenna": gaintable_mock.antenna.size,
                        "receptor1": gaintable_mock.receptor1.size,
                        "receptor2": gaintable_mock.receptor2.size,
                    }
                },
                kwargs={
                    "soln_time": 1,
                    "beams_factory": beams_factory,
                },
            ),
            call(
                _prediction_central_beams_ufunc,
                frequency_mock,
                input_core_dims=[[]],
                output_core_dims=[("antenna", "receptor1", "receptor2")],
                dask="parallelized",
                output_dtypes=[
                    np.complex128,
                ],
                join="outer",
                dataset_join="outer",
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "antenna": gaintable_mock.antenna.size,
                        "receptor1": gaintable_mock.receptor1.size,
                        "receptor2": gaintable_mock.receptor2.size,
                    }
                },
                kwargs={
                    "soln_time": 2,
                    "beams_factory": beams_factory,
                },
            ),
        ]
    )

    concat_mock.assert_called_once_with(
        ["TRANSPOSE_1", "TRANSPOSE_2"], dim="time"
    )

    transpose_mock.assert_has_calls(
        [
            call("antenna", "frequency", "receptor1", "receptor2"),
            call("antenna", "frequency", "receptor1", "receptor2"),
        ]
    )

    response_mock.assign_attrs.assert_called_once_with(
        gaintable_mock.gain.attrs
    )
    response_mock.assign_coords.assert_called_once_with(
        gaintable_mock.gain.coords
    )
    gaintable_mock.assign.assert_called_once_with({"gain": response_mock})
    assert expected == gaintable_mock.assign.return_value
