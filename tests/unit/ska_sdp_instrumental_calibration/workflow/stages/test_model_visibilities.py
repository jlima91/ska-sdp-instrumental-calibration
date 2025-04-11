from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import predict_vis_stage


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".get_phasecentre"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".generate_lsm_from_gleamegc"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.model_visibilities"
    ".predict_vis"
)
def test_should_predict_visibilities(
    predict_vis_mock, generate_lsm_mock, get_phasecentre_mock
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="Visibilities")
    cli_args = {"input": "path/to/input/ms"}
    get_phasecentre_mock.return_value = (0.0, 0.0)
    generate_lsm_mock.return_value = ["source1", "source2"]
    predict_vis_mock.return_value = [1, 2, 3]

    params = {
        "beam_type": "everybeam",
        "eb_ms": "test.ms",
        "eb_coeffs": "/path/to/coeffs",
        "gleamfile": "/path/to/gleam.dat",
        "fov": 10.0,
        "flux_limit": 1.0,
        "export_model_vis": False,
    }

    result = predict_vis_stage.stage_definition(
        upstream_output, **params, _cli_args_=cli_args
    )

    get_phasecentre_mock.assert_called_once_with("path/to/input/ms")
    generate_lsm_mock.assert_called_once_with(
        gleamfile="/path/to/gleam.dat",
        phasecentre=(0.0, 0.0),
        fov=10.0,
        flux_limit=1.0,
    )
    predict_vis_mock.assert_called_once_with(
        upstream_output.vis,
        ["source1", "source2"],
        beam_type="everybeam",
        eb_ms="test.ms",
        eb_coeffs="/path/to/coeffs",
    )

    assert result.modelvis == [1, 2, 3]
