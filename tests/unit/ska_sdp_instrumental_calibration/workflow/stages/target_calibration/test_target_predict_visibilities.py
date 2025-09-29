from mock import Mock, patch

from ska_sdp_instrumental_calibration.workflow.stages import target_calibration

predict_vis_stage = target_calibration.predict_vis_stage


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.target_calibration"
    ".predict_visibilities.predict_visibilities"
)
def test_should_predict_visibilities(predict_visibilities_mock):

    upstream_output = Mock(name="UpstreamOutput")
    cli_args = {"input": "path/to/input/ms"}
    predict_visibilities_mock.return_value = upstream_output

    params = {
        "beam_type": "everybeam",
        "eb_ms": None,
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": "/path/to/gleam.dat",
        "lsm_csv_path": None,
        "fov": 10.0,
        "flux_limit": 1.0,
        "alpha0": -0.78,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    predict_visibilities_mock.assert_called_once_with(
        upstream_output=upstream_output,
        beam_type="everybeam",
        normalise_at_beam_centre=False,
        eb_ms=None,
        eb_coeffs="/path/to/coeffs",
        gleamfile="/path/to/gleam.dat",
        lsm_csv_path=None,
        fov=10.0,
        flux_limit=1.0,
        alpha0=-0.78,
        _cli_args_=cli_args,
    )

    assert result == predict_visibilities_mock.return_value
