from ska_sdp_instrumental_calibration.stages import target_calibration

predict_vis_stage = target_calibration.predict_vis_stage


def test_should_predict_visibilities():

    expected_config = {
        "predict_vis": {
            "use_everybeam": True,
            "normalise_at_beam_centre": True,
            "eb_ms": None,
            "element_response_model": "oskar_dipole_cos",
            "gleamfile": None,
            "lsm_csv_path": None,
            "fov": 5.0,
            "flux_limit": 1.0,
            "alpha0": -0.78,
            "export_sky_model": True,
        }
    }

    assert predict_vis_stage.__stage__.config == expected_config
