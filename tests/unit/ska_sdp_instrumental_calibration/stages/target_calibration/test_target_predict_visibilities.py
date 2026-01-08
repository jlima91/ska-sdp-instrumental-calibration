from ska_sdp_instrumental_calibration.stages import target_calibration

predict_vis_stage = target_calibration.predict_vis_stage


def test_should_predict_visibilities():

    expected_params = [
        "upstream_output",
        "beam_type",
        "normalise_at_beam_centre",
        "eb_ms",
        "gleamfile",
        "lsm_csv_path",
        "element_response_model",
        "fov",
        "flux_limit",
        "alpha0",
        "_cli_args_",
    ]

    assert predict_vis_stage.stage_definition_args == expected_params
